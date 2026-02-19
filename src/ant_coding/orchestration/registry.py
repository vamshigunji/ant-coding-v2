"""
Registry for orchestration patterns with decorator-based registration.
"""

from typing import Dict, List, Type

from ant_coding.orchestration.base import OrchestrationPattern


class PatternNotFoundError(Exception):
    """Raised when a requested pattern is not registered."""

    pass


class DuplicatePatternError(Exception):
    """Raised when a pattern with the same name is already registered."""

    pass


class OrchestrationRegistry:
    """
    Central registry for orchestration patterns.

    Patterns register themselves via the @OrchestrationRegistry.register decorator
    and can be retrieved by name.
    """

    _registry: Dict[str, Type[OrchestrationPattern]] = {}

    @classmethod
    def register(cls, pattern_cls: Type[OrchestrationPattern]) -> Type[OrchestrationPattern]:
        """
        Decorator to register an orchestration pattern class.

        Args:
            pattern_cls: The OrchestrationPattern subclass to register.

        Returns:
            The same class, unmodified.

        Raises:
            DuplicatePatternError: If a pattern with the same name is already registered.
        """
        instance = pattern_cls()
        name = instance.name()
        if name in cls._registry:
            raise DuplicatePatternError(
                f"Pattern '{name}' is already registered by {cls._registry[name].__name__}"
            )
        cls._registry[name] = pattern_cls
        return pattern_cls

    @classmethod
    def get(cls, name: str) -> OrchestrationPattern:
        """
        Retrieve a registered pattern by name.

        Args:
            name: The pattern name.

        Returns:
            A new instance of the registered pattern.

        Raises:
            PatternNotFoundError: If no pattern with that name is registered.
        """
        if name not in cls._registry:
            raise PatternNotFoundError(
                f"Pattern '{name}' not found. Available: {cls.list_available()}"
            )
        return cls._registry[name]()

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered pattern names.

        Returns:
            Sorted list of pattern names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._registry.clear()
