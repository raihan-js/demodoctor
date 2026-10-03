"""DemoDoctor: find bad robot demonstrations, measure what they cost a policy."""

from demodoctor.faults import FAULTS, FaultManifest, inject_fault

__version__ = "0.1.0"
__all__ = ["FAULTS", "FaultManifest", "inject_fault"]