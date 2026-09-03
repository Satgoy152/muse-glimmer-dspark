#!/usr/bin/env python3
"""Trace generation with a sliding window of instance images.

Images are ~6 GB each after layer dedup, so all 2000 is ~12 TB against a 1 TB
volume and they cannot all be resident. A janitor thread pulls ahead of the
frontier and drops each image once its trajectory lands.

One harness process per effort segment, so mini-swe-agent keeps ownership of
concurrency and of preds.json, which it read-modify-writes and which would
corrupt under parallel writers.

Images are removed by explicit name and only for finished instances. A blanket
`docker image prune -a` on a timer races in-flight pulls: a w=30 run lost 11/50
instances to `docker run` exiting 125/127 under such a loop.
"""
import argparse, json, os, shutil, signal, socket, subprocess, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EFFORT_PLAN = [(0, 600, "low"), (600, 1200, "medium"), (1200, 1800, "high"), (1800, 2000, "xhigh")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb(path="/data"):
    return shutil.disk_usage(path).free / 1024**3


def pull(image):
    r = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True)
    if r.returncode:
        log(f"  pull FAILED {image}: {r.stderr.strip()[:160]}")
    return r.returncode == 0


def resident_images():
    r = subprocess.run(["docker", "images", "--format", "{{.Repository}}"],
                       capture_output=True, text=True)
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def healthy(url, token, timeout=10):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        return urllib.request.urlopen(req, timeout=timeout).status == 200
    except Exception:
        return False


def stop_orphan_containers():
    """Stop task containers left behind when the harness exits.

    mini-swe-agent starts each task container running `sleep 2h`, so `--rm` only
    fires when that sleep ends. Kill the harness -- as the watchdog does on a
    preemption -- and its containers keep running for up to two hours, each
    pinning its image so the janitor cannot reclaim the space. Measured after
    one watchdog trip: 66 containers alive against 20 workers, holding 320 GB
    and pushing free space under the prefetch threshold.

    Only ever called with no harness running, so every match is an orphan.
    """
    ids = subprocess.run(["docker", "ps", "-q", "--filter", "name=minisweagent-"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        log(f"  stopping {len(ids)} orphaned task containers")
        subprocess.run(["docker", "stop", "-t", "5", *ids], capture_output=True)
    subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)


def proxy_alive(hostport):
    """TCP connect to the recording proxy.

    The harness talks to the proxy, not to the endpoint, so a healthy endpoint
    says nothing about whether calls can actually be made. A dead proxy shows up
    as litellm InternalServerError, which the harness treats as a failed
    instance and writes to preds.json -- silently burning it.
    """
    if not hostport:
        return True
    host, _, port = hostport.rpartition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=5):
            return True
    except Exception:
        return False


def wait_for_endpoint(url, token, interval=30):
    if not url or healthy(url, token):
        return
    log(f"  endpoint unreachable, waiting for it to return: {url}")
    t0 = time.time()
    while not healthy(url, token):
        time.sleep(interval)
    log(f"  endpoint back after {time.time()-t0:.0f}s")


class Watchdog(threading.Thread):
    """Stop the harness when the endpoint goes away.

    These endpoints are preemptible. Left alone, the harness keeps handing
    instances to a dead upstream; litellm retries, gives up, and the instance
    lands in preds.json as done -- so it is skipped on resume and lost. Killing
    the harness instead leaves those instances simply unfinished, and the next
    pass picks them up.

    Requires several consecutive failures so a single 504 does not trip it.
    """

    def __init__(self, url, token, proc, proxy=None, fails=3, interval=30):
        super().__init__(daemon=True)
        self.url, self.token, self.proc, self.proxy = url, token, proc, proxy
        self.fails, self.interval = fails, interval
        self.stop, self.tripped = threading.Event(), False

    def run(self):
        consecutive = 0
        while not self.stop.is_set():
            if self.proc.poll() is not None:
                return
            ok_endpoint = healthy(self.url, self.token)
            ok_proxy = proxy_alive(self.proxy)
            if ok_endpoint and ok_proxy:
                consecutive = 0
            else:
                consecutive += 1
                what = "endpoint" if not ok_endpoint else "proxy"
                log(f"  watchdog: {what} unreachable ({consecutive}/{self.fails})")
                if consecutive >= self.fails:
                    self.tripped = True
                    log("  watchdog: stopping harness to avoid burning instances")
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    except Exception as e:
                        log(f"  watchdog: killpg failed: {e}")
                    return
            self.stop.wait(self.interval)


def in_use(image):
    r = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={image}"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


class Janitor(threading.Thread):
    """Keeps ~window images resident: pulls ahead of the frontier, drops finished."""

    def __init__(self, rows, lo, hi, out, window, pull_jobs, min_free, interval=10):
        super().__init__(daemon=True)
        self.rows, self.lo, self.hi, self.out = rows, lo, hi, out
        self.window, self.min_free, self.interval = window, min_free, interval
        self.pool = ThreadPoolExecutor(max_workers=pull_jobs)
        self.stop = threading.Event()
        self.staged, self.dropped = set(resident_images()), set()
        self.pulled = self.deleted = 0

    def done(self, j):
        iid = self.rows[j]["instance_id"]
        return (self.out / iid / f"{iid}.traj.json").exists()

    def prestage(self):
        """Pull the window and block, before the harness is allowed to start.

        Otherwise the harness fires `workers` `docker run`s at once, each with
        its own on-demand pull, competing with the janitor for the same volume.
        At -w 60 that saturated the disk at 100% util with an 82-deep queue and
        only 2 of 61 instances ran. Pulls are disk-bandwidth bound, so total
        pull concurrency must stay small regardless of worker count.
        """
        ahead = [j for j in range(self.lo, self.hi) if not self.done(j)][: self.window]
        todo = [self.rows[j]["image_name"] for j in ahead
                if self.rows[j]["image_name"] not in self.staged]
        self.staged.update(todo)
        if not todo:
            log("  prestage: window already warm")
            return
        log(f"  prestage: pulling {len(todo)} images before harness start")
        t0 = time.time()
        ok = sum(self.pool.map(pull, todo))
        self.pulled += len(todo)
        log(f"  prestage: {ok}/{len(todo)} in {time.time()-t0:.0f}s, free={free_gb():.0f}G")

    def run(self):
        while not self.stop.is_set():
            try:
                self.sweep()
            except Exception as e:  # a janitor crash must not take down the run
                log(f"  janitor error: {type(e).__name__}: {e}")
            self.stop.wait(self.interval)

    def sweep(self):
        # Reap containers first: one created but never started stays in `Created`,
        # `--rm` never fires, and it pins its image so the rm below fails.
        # Measured: 31 leaked containers pinned 265 GB. The age filter avoids
        # racing a container mid-creation.
        subprocess.run(["docker", "container", "prune", "-f", "--filter", "until=10m"],
                       capture_output=True)

        # Only mark an image dropped when removal actually succeeded -- a failed
        # rm must be retried on the next sweep, not forgotten.
        for j in range(self.lo, self.hi):
            img = self.rows[j]["image_name"]
            if img in self.dropped or not self.done(j) or in_use(img):
                continue
            r = subprocess.run(["docker", "image", "rm", img], capture_output=True)
            if r.returncode == 0:
                self.dropped.add(img)
                self.deleted += 1

        if free_gb() < self.min_free:
            log(f"  janitor: free={free_gb():.0f}G below {self.min_free}G, holding off pulls")
            return

        # mini-swe-agent hands work out roughly in slice order, so the next
        # unfinished indices are the ones about to run.
        ahead = [j for j in range(self.lo, self.hi) if not self.done(j)][: self.window]
        for j in ahead:
            img = self.rows[j]["image_name"]
            if img not in self.staged:
                self.staged.add(img)
                self.pool.submit(pull, img)
                self.pulled += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="data/training/swegym_2k")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="/data/runs/train")
    p.add_argument("--window", type=int, default=60, help="target resident images")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=2000)
    p.add_argument("--pull-jobs", type=int, default=8)
    p.add_argument("--min-free-gb", type=float, default=150)
    p.add_argument("--health", default=os.environ.get("UPSTREAM_HEALTH_URL", ""),
                   help="endpoint /v1/models URL; watchdog pauses the run when it fails")
    p.add_argument("--token", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--proxy", default=os.environ.get("PROXY_HOSTPORT", ""),
                   help="host:port of the recording proxy; watchdog pauses if it dies")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    rows = [json.loads(l) for l in open(Path(a.subset) / f"{a.split}.jsonl")]
    out = Path(a.out)

    for lo, hi, effort in EFFORT_PLAN:
        lo, hi = max(lo, a.start), min(hi, a.end)
        if lo >= hi:
            continue
        tag = f"[{lo}:{hi}] effort={effort}"
        if a.dry_run:
            log(f"{tag}: {hi-lo} instances, window={a.window}, -w {a.workers}")
            continue

        attempt = 0
        while True:
            attempt += 1
            jan = Janitor(rows, lo, hi, out, a.window, a.pull_jobs, a.min_free_gb)
            if all(jan.done(j) for j in range(lo, hi)):
                log(f"{tag}: complete")
                break
            wait_for_endpoint(a.health, a.token)   # no-op when --health is unset
            log(f"{tag}: starting (attempt {attempt}), window={a.window}, free={free_gb():.0f}G")
            jan.prestage()
            jan.start()
            t0 = time.time()
            proc = subprocess.Popen(
                ["uv", "run", "mini-extra", "swebench",
                 "--subset", a.subset, "--split", a.split, "--slice", f"{lo}:{hi}",
                 "-c", "swebench.yaml", "-c", "configs/swegym.yaml",
                 "-c", f"configs/effort_{effort}.yaml",
                 "-w", str(a.workers), "-o", str(out)],
                start_new_session=True)  # own process group, so the watchdog can kill it whole
            # Without a health URL there is nothing to watch, so the run behaves
            # exactly as it did before rather than tripping on every check.
            wd = Watchdog(a.health, a.token, proc, a.proxy) if (a.health or a.proxy) else None
            if wd:
                wd.start()
            rc = proc.wait()
            if wd:
                wd.stop.set()
            jan.stop.set()
            stop_orphan_containers()   # before the sweep, so images unpin first
            jan.join(timeout=30)
            jan.sweep()

            done = sum(jan.done(j) for j in range(lo, hi))
            log(f"{tag}: exit={rc} {done}/{hi-lo} trajectories in {time.time()-t0:.0f}s "
                f"(pulled {jan.pulled}, released {jan.deleted}, free={free_gb():.0f}G)")
            if not (wd and wd.tripped):
                break
            log(f"{tag}: harness stopped by watchdog, will resume when the endpoint returns")


if __name__ == "__main__":
    main()
