"""Copy of sensai.util.string from sensAI """
# From commit commit d7b4afcc89b4d2e922a816cb07dffde27f297354


import functools
import logging
import re
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    Any,
    Self,
)

reCommaWhitespacePotentiallyBreaks = re.compile(r",\s+")

log = logging.getLogger(__name__)

# ruff: noqa


class StringConverter(ABC):
    """Abstraction for a string conversion mechanism."""

    @abstractmethod
    def to_string(self, x: Any) -> str:
        pass


def dict_string(
    d: Mapping, brackets: str | None = None, converter: StringConverter | None = None
) -> str:
    """Converts a dictionary to a string of the form "<key>=<value>, <key>=<value>, ...", optionally enclosed
    by brackets.

    :param d: the dictionary
    :param brackets: a two-character string containing the opening and closing bracket to use, e.g. ``"{}"``;
        if None, do not use enclosing brackets
    :param converter: the string converter to use for values
    :return: the string representation
    """
    s = ", ".join([f"{k}={to_string(v, converter=converter, context=k)}" for k, v in d.items()])
    if brackets is not None:
        return brackets[:1] + s + brackets[-1:]
    else:
        return s


def list_string(
    l: Iterable[Any],
    brackets: str | None = "[]",
    quote: str | None = None,
    converter: StringConverter | None = None,
) -> str:
    """Converts a list or any other iterable to a string of the form "[<value>, <value>, ...]", optionally enclosed
    by different brackets or with the values quoted.

    :param l: the list
    :param brackets: a two-character string containing the opening and closing bracket to use, e.g. ``"[]"``;
        if None, do not use enclosing brackets
    :param quote: a 1-character string defining the quote to use around each value, e.g. ``"'"``.
    :param converter: the string converter to use for values
    :return: the string representation
    """

    def item(x: Any) -> str:
        x = to_string(x, converter=converter, context="list")
        if quote is not None:
            return quote + x + quote
        else:
            return x

    s = ", ".join(item(x) for x in l)
    if brackets is not None:
        return brackets[:1] + s + brackets[-1:]
    else:
        return s


def to_string(
    x: Any,
    converter: StringConverter | None = None,
    apply_converter_to_non_complex_objects: bool = True,
    context: Any = None,
) -> str:
    """Converts the given object to a string, with proper handling of lists, tuples and dictionaries, optionally using a converter.
    The conversion also removes unwanted line breaks (as present, in particular, in sklearn's string representations).

    :param x: the object to convert
    :param converter: the converter with which to convert objects to strings
    :param apply_converter_to_non_complex_objects: whether to apply/pass on the converter (if any) not only when converting complex objects
        but also non-complex, primitive objects; use of this flag enables converters to implement their conversion functionality using this
        function for complex objects without causing an infinite recursion.
    :param context: context in which the object is being converted (e.g. dictionary key for case where x is the corresponding
        dictionary value), only for debugging purposes (will be reported in log messages upon recursion exception)
    :return: the string representation
    """
    try:
        if isinstance(x, list):
            return list_string(x, converter=converter)
        elif isinstance(x, tuple):
            return list_string(x, brackets="()", converter=converter)
        elif isinstance(x, dict):
            return dict_string(x, brackets="{}", converter=converter)
        elif isinstance(x, types.MethodType):
            # could be bound method of a ToStringMixin instance (which would print the repr of the instance, which can potentially cause
            # an infinite recursion)
            return f"Method[{x.__name__}]"
        else:
            if converter and apply_converter_to_non_complex_objects:
                s = converter.to_string(x)
            else:
                s = str(x)

            # remove any unwanted line breaks and indentation after commas (as generated, for example, by sklearn objects)
            return reCommaWhitespacePotentiallyBreaks.sub(", ", s)

    except RecursionError:
        log.error(f"Recursion in string conversion detected; context={context}")
        raise


def object_repr(obj: Any, member_names_or_dict: list[str] | dict[str, Any]) -> str:
    """Creates a string representation for the given object based on the given members.

    The string takes the form "ClassName[attr1=value1, attr2=value2, ...]"
    """
    if isinstance(member_names_or_dict, dict):
        members_dict = member_names_or_dict
    else:
        members_dict = {m: to_string(getattr(obj, m)) for m in member_names_or_dict}
    return f"{obj.__class__.__name__}[{dict_string(members_dict)}]"


def or_regex_group(allowed_names: Sequence[str]) -> str:
    """:param allowed_names: strings to include as literals in the regex
    :return: a regular expression string of the form `(<name_1>| ...|<name_N>)`, which any of the given names
    """
    allowed_names = [re.escape(name) for name in allowed_names]
    return r"(%s)" % "|".join(allowed_names)


def function_name(x: Callable) -> str:
    """Attempts to retrieve the name of the given function/callable object, taking the possibility
    of the function being defined via functools.partial into account.

    :param x: a callable object
    :return: name of the function or str(x) as a fallback
    """
    if isinstance(x, functools.partial):
        return function_name(x.func)
    elif hasattr(x, "__name__"):
        return x.__name__
    else:
        return str(x)


class ToStringMixin:
    """Provides implementations for ``__str__`` and ``__repr__`` which are based on the format ``"<class name>[<object info>]"`` and
    ``"<class name>[id=<object id>, <object info>]"`` respectively, where ``<object info>`` is usually a list of entries of the
    form ``"<name>=<value>, ..."``.

    By default, ``<class name>`` will be the qualified name of the class, and ``<object info>`` will include all properties
    of the class, including private ones starting with an underscore (though the underscore will be dropped in the string
    representation).

        * To exclude private properties, override :meth:`_toStringExcludePrivate` to return True. If there are exceptions
          (and some private properties shall be retained), additionally override :meth:`_toStringExcludeExceptions`.
        * To exclude a particular set of properties, override :meth:`_toStringExcludes`.
        * To include only select properties (introducing inclusion semantics), override :meth:`_toStringIncludes`.
        * To add values to the properties list that aren't actually properties of the object (i.e. derived properties),
          override :meth:`_toStringAdditionalEntries`.
        * To define a fully custom representation for ``<object info>`` which is not based on the above principles, override
          :meth:`_toStringObjectInfo`.

    For well-defined string conversions within a class hierarchy, it can be a good practice to define additional
    inclusions/exclusions by overriding the respective method once more and basing the return value on an extended
    version of the value returned by superclass.
    In some cases, the requirements of a subclass can be at odds with the definitions in the superclass: The superclass
    may make use of exclusion semantics, but the subclass may want to use inclusion semantics (and include
    only some of the many properties it adds). In this case, if the subclass used :meth:`_toStringInclude`, the exclusion semantics
    of the superclass would be void and none of its properties would actually be included.
    In such cases, override :meth:`_toStringIncludesForced` to add inclusions regardless of the semantics otherwise used along
    the class hierarchy.

    """

    _TOSTRING_INCLUDE_ALL = "__all__"

    def _tostring_class_name(self) -> str:
        """:return: the string use for <class name> in the string representation ``"<class name>[<object info]"``"""
        return type(self).__qualname__

    def _tostring_properties(
        self,
        exclude: str | Iterable[str] | None = None,
        include: str | Iterable[str] | None = None,
        exclude_exceptions: list[str] | None = None,
        include_forced: list[str] | None = None,
        additional_entries: dict[str, Any] | None = None,
        converter: StringConverter | None = None,
    ) -> str:
        """Creates a string of the class attributes, with optional exclusions/inclusions/additions.
        Exclusions take precedence over inclusions.

        :param exclude: attributes to be excluded
        :param include: attributes to be included; if non-empty, only the specified attributes will be printed (bar the ones
            excluded by ``exclude``)
        :param include_forced: additional attributes to be included
        :param additional_entries: additional key-value entries to be added
        :param converter: the string converter to use; if None, use default (which avoids infinite recursions)
        :return: a string containing entry/property names and values
        """

        def mklist(x: Any) -> list[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return x

        exclude = mklist(exclude)
        include = mklist(include)
        include_forced = mklist(include_forced)
        exclude_exceptions = mklist(exclude_exceptions)

        def is_excluded(k: Any) -> bool:
            if k in include_forced or k in exclude_exceptions:
                return False
            if k in exclude:
                return True
            if self._tostring_exclude_private():
                return k.startswith("_")
            else:
                return False

        # determine relevant attribute dictionary
        if (
            len(include) == 1 and include[0] == self._TOSTRING_INCLUDE_ALL
        ):  # exclude semantics (include everything by default)
            attribute_dict = self.__dict__
        else:  # include semantics (include only inclusions)
            attribute_dict = {
                k: getattr(self, k)
                for k in set(include + include_forced)
                if hasattr(self, k) and k != self._TOSTRING_INCLUDE_ALL
            }

        # apply exclusions and remove underscores from attribute names
        d = {k.strip("_"): v for k, v in attribute_dict.items() if not is_excluded(k)}

        if additional_entries is not None:
            d.update(additional_entries)

        if converter is None:
            converter = self._StringConverterAvoidToStringMixinRecursion(self)
        return dict_string(d, converter=converter)

    def _tostring_object_info(self) -> str:
        """Override this method to use a fully custom definition of the ``<object info>`` part in the full string
        representation ``"<class name>[<object info>]"`` to be generated.
        As soon as this method is overridden, any property-based exclusions, inclusions, etc. will have no effect
        (unless the implementation is specifically designed to make use of them - as is the default
        implementation).
        NOTE: Overrides must not internally use super() because of a technical limitation in the proxy
        object that is used for nested object structures.

        :return: a string containing the string to use for ``<object info>``
        """
        return self._tostring_properties(
            exclude=self._tostring_excludes(),
            include=self._tostring_includes(),
            exclude_exceptions=self._tostring_exclude_exceptions(),
            include_forced=self._tostring_includes_forced(),
            additional_entries=self._tostring_additional_entries(),
        )

    def _tostring_excludes(self) -> list[str]:
        """Makes the string representation exclude the returned attributes.
        This method can be conveniently overridden by subclasses which can call super and extend the list returned.

        This method will only have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names
        """
        return []

    def _tostring_includes(self) -> list[str]:
        """Makes the string representation include only the returned attributes (i.e. introduces inclusion semantics);
        By default, the list contains only a marker element, which is interpreted as "all attributes included".

        This method can be conveniently overridden by sub-classes which can call super and extend the list returned.
        Note that it is not a problem for a list containing the aforementioned marker element (which stands for all attributes)
        to be extended; the marker element will be ignored and only the user-added elements will be considered as included.

        Note: To add an included attribute in a sub-class, regardless of any super-classes using exclusion or inclusion semantics,
        use _toStringIncludesForced instead.

        This method will have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names to be included in the string representation
        """
        return [self._TOSTRING_INCLUDE_ALL]

    # noinspection PyMethodMayBeStatic
    def _tostring_includes_forced(self) -> list[str]:
        """Defines a list of attribute names that are required to be present in the string representation, regardless of the
        instance using include semantics or exclude semantics, thus facilitating added inclusions in sub-classes.

        This method will have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names
        """
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        """:return: a dictionary of entries to be included in the ``<object info>`` part of the string representation"""
        return {}

    def _tostring_exclude_private(self) -> bool:
        """:return: whether to exclude properties that are private (start with an underscore); explicitly included attributes
        will still be considered - as will properties exempt from the rule via :meth:`toStringExcludeException`.
        """
        return False

    def _tostring_exclude_exceptions(self) -> list[str]:
        """Defines attribute names which should not be excluded even though other rules (particularly the exclusion of private members
        via :meth:`_toStringExcludePrivate`) would otherwise exclude them.

        :return: a list of attribute names
        """
        return []

    def __str__(self) -> str:
        return f"{self._tostring_class_name()}[{self._tostring_object_info()}]"

    def __repr__(self) -> str:
        info = f"id={id(self)}"
        property_info = self._tostring_object_info()
        if len(property_info) > 0:
            info += ", " + property_info
        return f"{self._tostring_class_name()}[{info}]"

    def pprint(self, file: Any = sys.stdout) -> None:
        """Prints a prettily formatted string representation of the object (with line breaks and indentations)
        to ``stdout`` or the given file.

        :param file: the file to print to
        """
        print(self.pprints(), file=file)

    def pprints(self) -> str:
        """:return: a prettily formatted string representation with line breaks and indentations"""
        return pretty_string_repr(self)

    class _StringConverterAvoidToStringMixinRecursion(StringConverter):
        """Avoids recursions when converting objects implementing :class:`ToStringMixin` which may contain themselves to strings.
        Use of this object prevents infinite recursions caused by a :class:`ToStringMixin` instance recursively containing itself in
        either a property of another :class:`ToStringMixin`, a list or a tuple.
        It handles all :class:`ToStringMixin` instances recursively encountered.

        A previously handled instance is converted to a string of the form "<class name>[<<]".
        """

        def __init__(self, *handled_objects: "ToStringMixin"):
            """:param handled_objects: objects which are initially assumed to have been handled already"""
            self._handled_to_string_mixin_ids = {id(o) for o in handled_objects}

        def to_string(self, x: Any) -> str:
            if isinstance(x, ToStringMixin):
                oid = id(x)
                if oid in self._handled_to_string_mixin_ids:
                    return f"{x._tostring_class_name()}[<<]"
                self._handled_to_string_mixin_ids.add(oid)
                return str(self._ToStringMixinProxy(x, self))
            else:
                return to_string(
                    x,
                    converter=self,
                    apply_converter_to_non_complex_objects=False,
                    context=x.__class__,
                )

        class _ToStringMixinProxy:
            """A proxy object which wraps a ToStringMixin to ensure that the converter is applied when creating the properties string.
            The proxy is to achieve that all ToStringMixin methods that aren't explicitly overwritten are bound to this proxy
            (rather than the original object), such that the transitive call to _toStringProperties will call the new
            implementation.
            """

            # methods where we assume that they could transitively call _toStringProperties (others are assumed not to)
            TOSTRING_METHODS_TRANSITIVELY_CALLING_TOSTRINGPROPERTIES = {"_tostring_object_info"}

            def __init__(self, x: "ToStringMixin", converter: Any) -> None:
                self.x = x
                self.converter = converter

            def _tostring_properties(self, *args: Any, **kwargs: Any) -> str:
                return self.x._tostring_properties(*args, **kwargs, converter=self.converter)  # type: ignore[misc]

            def _tostring_class_name(self) -> str:
                return self.x._tostring_class_name()

            def __getattr__(self, attr: str) -> Any:
                if attr.startswith(
                    "_tostring",
                ):  # ToStringMixin method which we may bind to use this proxy to ensure correct transitive call
                    method = getattr(self.x.__class__, attr)
                    obj = (
                        self
                        if attr in self.TOSTRING_METHODS_TRANSITIVELY_CALLING_TOSTRINGPROPERTIES
                        else self.x
                    )
                    return lambda *args, **kwargs: method(obj, *args, **kwargs)
                else:
                    return getattr(self.x, attr)

            def __str__(self) -> str:
                return ToStringMixin.__str__(self)  # type: ignore[arg-type]


def pretty_string_repr(
    s: Any, initial_indentation_level: int = 0, indentation_string: str = "    "
) -> str:
    """Creates a pretty string representation (using indentations) from the given object/string representation (as generated, for example, via
    ToStringMixin). An indentation level is added for every opening bracket.

    :param s: an object or object string representation
    :param initial_indentation_level: the initial indentation level
    :param indentation_string: the string which corresponds to a single indentation level
    :return: a reformatted version of the input string with added indentations and line breaks
    """
    if not isinstance(s, str):
        s = str(s)
    indent = initial_indentation_level
    result = indentation_string * indent
    i = 0

    def nl() -> None:
        nonlocal result
        result += "\n" + (indentation_string * indent)

    def take(cnt: int = 1) -> None:
        nonlocal result, i
        result += s[i : i + cnt]
        i += cnt

    def find_matching(j: int) -> int | None:
        start = j
        op = s[j]
        cl = {"[": "]", "(": ")", "'": "'"}[s[j]]
        is_bracket = cl != s[j]
        stack = 0
        while j < len(s):
            if s[j] == op and (is_bracket or j == start):
                stack += 1
            elif s[j] == cl:
                stack -= 1
            if stack == 0:
                return j
            j += 1
        return None

    brackets = "[("
    quotes = "'"
    while i < len(s):
        is_bracket = s[i] in brackets
        is_quote = s[i] in quotes
        if is_bracket or is_quote:
            i_match = find_matching(i)
            take_full_match_without_break = False
            if i_match is not None:
                k = i_match + 1
                full_match = s[i:k]
                take_full_match_without_break = is_quote or not (
                    "=" in full_match and "," in full_match
                )
                if take_full_match_without_break:
                    take(k - i)
            if not take_full_match_without_break:
                take(1)
                indent += 1
                nl()
        elif s[i] in "])":
            take(1)
            indent -= 1
        elif s[i : i + 2] == ", ":
            take(2)
            nl()
        else:
            take(1)

    return result


class TagBuilder:
    """Assists in building strings made up of components that are joined via a glue string."""

    def __init__(self, *initial_components: str, glue: str = "_"):
        """:param initial_components: initial components to always include at the beginning
        :param glue: the glue string which joins components
        """
        self.glue = glue
        self.components = list(initial_components)

    def with_component(self, component: str) -> Self:
        self.components.append(component)
        return self

    def with_conditional(self, cond: bool, component: str) -> Self:
        """Conditionally adds the given component.

        :param cond: the condition
        :param component: the component to add if the condition holds
        :return: the builder
        """
        if cond:
            self.components.append(component)
        return self

    def with_alternative(self, cond: bool, true_component: str, false_component: str) -> Self:
        """Adds a component depending on a condition.

        :param cond: the condition
        :param true_component: the component to add if the condition holds
        :param false_component: the component to add if the condition does not hold
        :return: the builder
        """
        self.components.append(true_component if cond else false_component)
        return self

    def build(self) -> str:
        """:return: the string (with all components joined)"""
        return self.glue.join(self.components)
