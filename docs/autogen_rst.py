import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(os.path.basename(__file__))


def module_template(module_qualname: str):
    module_name = module_qualname.split(".")[-1]
    title = module_name.replace("_", r"\_")
    return f"""{title}
{"="*len(title)}

.. automodule:: {module_qualname}
   :members:
   :undoc-members:
"""


def package_template(package_qualname: str):
    package_name = package_qualname.split(".")[-1]
    title = package_name.replace("_", r"\_")
    return f"""{title}
{"="*len(title)}

.. automodule:: {package_qualname}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   {package_name}/*
"""


def indexTemplate(package_name):
    title = package_name
    return f"""{title}
{"="*len(title)}

.. automodule:: {package_name}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   *
"""


def write_to_file(content: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o666)


def make_rst(src_root, rst_root, clean=False, overwrite=False, package_prefix=""):
    """Creates/updates documentation in form of rst files for modules and packages.
    Does not delete any existing rst files. Thus, rst files for packages or modules that have been removed or renamed
    should be deleted by hand.

    This method should be executed from the project's top-level directory

    :param src_root: path to library base directory, typically "src/<library_name>"
    :param clean: whether to completely clean the target directory beforehand, removing any existing .rst files
    :param overwrite: whether to overwrite existing rst files. This should be used with caution as it will delete
        all manual changes to documentation files
    :package_prefix: a prefix to prepend to each module (for the case where the src_root is not the base package),
        which, if not empty, should end with a "."
    :return:
    """
    rst_root = os.path.abspath(rst_root)

    if clean and os.path.isdir(rst_root):
        shutil.rmtree(rst_root)

    base_package_name = package_prefix + os.path.basename(src_root)
    write_to_file(indexTemplate(base_package_name), os.path.join(rst_root, "index.rst"))

    for root, dirnames, filenames in os.walk(src_root):
        if os.path.basename(root).startswith("_"):
            continue
        base_package_relpath = os.path.relpath(root, start=src_root)
        base_package_qualname = package_prefix + os.path.relpath(
            root,
            start=os.path.dirname(src_root),
        ).replace(os.path.sep, ".")

        for dirname in dirnames:
            if not dirname.startswith("_"):
                package_qualname = f"{base_package_qualname}.{dirname}"
                package_rst_path = os.path.join(rst_root, base_package_relpath, f"{dirname}.rst")
                log.info(f"Writing package documentation to {package_rst_path}")
                write_to_file(package_template(package_qualname), package_rst_path)

        for filename in filenames:
            base_name, ext = os.path.splitext(filename)
            if ext == ".py" and not filename.startswith("_"):
                module_qualname = f"{base_package_qualname}.{filename[:-3]}"

                module_rst_path = os.path.join(rst_root, base_package_relpath, f"{base_name}.rst")
                if os.path.exists(module_rst_path) and not overwrite:
                    log.debug(f"{module_rst_path} already exists, skipping it")

                log.info(f"Writing module documentation to {module_rst_path}")
                write_to_file(module_template(module_qualname), module_rst_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs_root = Path(__file__).parent
    make_rst(
        docs_root / ".." / "tianshou" / "highlevel",
        docs_root / "api" / "tianshou.highlevel",
        clean=True,
        package_prefix="tianshou.",
    )
