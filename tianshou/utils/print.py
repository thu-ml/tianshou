import pprint
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass
class DataclassPPrintMixin:
    def pprint_asdict(self, exclude_fields: Sequence[str] | None = None, indent: int = 4) -> None:
        """Pretty-print the object as a dict, excluding specified fields.

        :param exclude_fields: A sequence of field names to exclude from the output.
            If None, no fields are excluded.
        :param indent: The indentation to use when pretty-printing.
        """
        print(self.pprints_asdict(exclude_fields=exclude_fields, indent=indent))

    def pprints_asdict(self, exclude_fields: Sequence[str] | None = None, indent: int = 4) -> str:
        """String corresponding to pretty-print of the object as a dict, excluding specified fields.

        :param exclude_fields: A sequence of field names to exclude from the output.
            If None, no fields are excluded.
        :param indent: The indentation to use when pretty-printing.
        """
        prefix = f"{self.__class__.__name__}\n----------------------------------------\n"
        print_dict = asdict(self)
        exclude_fields = exclude_fields or []
        for field in exclude_fields:
            print_dict.pop(field, None)
        return prefix + pprint.pformat(print_dict, indent=indent)
