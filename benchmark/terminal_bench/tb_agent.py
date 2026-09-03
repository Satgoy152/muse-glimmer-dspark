#!/usr/bin/env python3
"""Terminal-Bench agent that routes mini-swe-agent through the recording proxy.

Terminal-Bench installs mini-swe-agent *inside* the task container and runs
`mini -m MODEL -t INSTRUCTION -y --exit-immediately`. Three consequences, each
of which would otherwise produce traces that do not match serve time:

  * The container cannot reach a proxy on the host loopback, and
    `host.docker.internal` does not resolve there -- tasks come up under
    per-task docker-compose, so there is no `extra_hosts` to set. The docker
    bridge gateway (172.17.0.1) is reachable with no host configuration.
  * The stock command passes no sampling parameters, so the run falls back to
    litellm defaults and loses top_k and reasoning_strength. The chat template
    defaults reasoning_strength to "high", so an unset run is not neutral.
  * The stock install template pins nothing and trusts the image's Python. See
    mini-swe-setup.sh.j2, which sits next to this file and is picked up because
    `_get_templated_script_path` resolves relative to the agent's own module.

mini's `-c` takes key=value specs whose values go through `json.loads`, so
sampling can be pushed on the command line rather than written into the
container. `-c mini.yaml` must come first: naming any config replaces the
default instead of adding to it.

Select with `--agent-import-path tb_agent:MuseGlimmerMiniAgent`.
"""

import json
import os
import shlex

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.agents.installed_agents.mini_swe_agent.mini_swe_agent import (
    MiniSweAgent,
)
from terminal_bench.terminal.models import TerminalCommand

MODEL = os.environ.get("TB_MODEL", "openai/meta-models/Muse-Glimmer-30B")
API_BASE = os.environ.get("TB_API_BASE", "http://172.17.0.1:8081/v1")
STRENGTH = os.environ.get("TB_REASONING_STRENGTH", "low")
STEP_LIMIT = os.environ.get("TB_STEP_LIMIT", "100")


class MuseGlimmerMiniAgent(MiniSweAgent):
    def __init__(self, model_name: str = MODEL, *args, **kwargs):
        # MiniSweAgent.__init__ does `provider, _ = model_name.split("/")`, which
        # raises on a three-segment name like openai/meta-models/Muse-Glimmer-30B.
        AbstractInstalledAgent.__init__(self, *args, **kwargs)
        self._model_name = model_name or MODEL
        self._provider = self._model_name.split("/", 1)[0]

    @property
    def _env(self) -> dict[str, str]:
        # The proxy needs no auth of its own, but mini and litellm both want a
        # key to be present before they will build a request.
        return {
            "MSWEA_CONFIGURED": "true",
            "MSWEA_API_KEY": "proxy",
            "OPENAI_API_KEY": "proxy",
        }

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        extra_body = json.dumps(
            {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": STRENGTH}},
            separators=(",", ":"),
        )
        specs = [
            "mini.yaml",
            f"model.model_kwargs.api_base={API_BASE}",
            "model.model_kwargs.api_key=proxy",
            "model.model_kwargs.temperature=1.0",
            "model.model_kwargs.top_p=0.95",
            "model.model_kwargs.drop_params=true",
            "model.model_kwargs.parallel_tool_calls=true",
            f"model.model_kwargs.extra_body={extra_body}",
            "model.cost_tracking=ignore_errors",
            f"agent.step_limit={STEP_LIMIT}",
        ]
        cfg = " ".join(f"-c {shlex.quote(s)}" for s in specs)
        return [
            TerminalCommand(
                command=(
                    f"mini -m {self._model_name} -t {shlex.quote(instruction)} "
                    f"-y --exit-immediately {cfg}"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
