"""Domain exceptions translated into safe API errors."""


class AgentError(RuntimeError):
    """Base class for expected Agent workflow errors."""


class IncidentNotFound(AgentError):
    pass


class NoActiveFault(AgentError):
    pass


class InvalidTransition(AgentError):
    pass


class PolicyViolation(AgentError):
    pass
