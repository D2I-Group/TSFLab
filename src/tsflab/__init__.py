"""TSFLab — Agent-first time-series forecasting framework.

Modules (one per stage of the research chain, mirrored by the `tsf` CLI):
    tsflab.core         contracts and schemas (pydantic only)
    tsflab.catalog      cards, registries, components, verification (`tsf catalog`, `tsf model`)
    tsflab.data         dataset loaders, profiles, preparation (`tsf data`)
    tsflab.models       flat model catalog, shared components, slot adapters
    tsflab.experiments  configuration, runner, losses, evaluation, execution (`tsf run`, `tsf env`)
    tsflab.release      Hub weights and data, submissions, leaderboard (`tsf result`)
    tsflab.realtime     rolling real-time tracks (`tsf realtime`)
    tsflab.research     research rounds and recombination (`tsf research`)
    tsflab.agent        agent assets, task templates, project scaffolding (`tsf agent`, `tsf init`)
    tsflab.cli          the `tsf` command line
    tsflab.assets       read-only resources bundled into wheels
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsflab")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0"
