from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pprint import pprint


@dataclass
class DataclassPPrintMixin:
    def pprint_asdict(self, exclude_fields: Sequence[str] | None = None) -> None:
        """Pretty-print the object as a dict, excluding specified fields.

        :param exclude_fields: A sequence of field names to exclude from the output.
            If None, no fields are excluded.
        """
        print(f"{self.__class__.__name__}")
        print("----------------------------------------")
        print_dict = asdict(self)
        exclude_fields = exclude_fields or []
        for field in exclude_fields:
            print_dict.pop(field, None)
        pprint(print_dict)
