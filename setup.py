from setuptools import setup

import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("openaerostruct/__init__.py").read(),
)[0]

optional_dependencies = {"docs": ["openmdao[docs]>=3.2, <=3.9.2"], "test": ["testflo>=1.3.6"]}

# Add an optional dependency that concatenates all others
optional_dependencies["all"] = sorted(
    [dependency for dependencies in optional_dependencies.values() for dependency in dependencies]
)

setup(
    name="openaerostruct",
    version=__version__,
    description="OpenAeroStruct",
    author="John Jasa",
    author_email="johnjasa@umich.edu",
    license="BSD-3",
    packages=[
        "openaerostruct",
        "openaerostruct/docs",
        "openaerostruct/docs/_utils",
        "openaerostruct/geometry",
        "openaerostruct/structures",
        "openaerostruct/aerodynamics",
        "openaerostruct/transfer",
        "openaerostruct/functionals",
        "openaerostruct/integration",
        "openaerostruct/common",
        "openaerostruct/utils",
    ],
    # Test files
    package_data={"openaerostruct": ["tests/*.py", "*/tests/*.py", "*/*/tests/*.py"]},
    install_requires=[
        "openmdao>=3.2",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require=optional_dependencies,
    zip_safe=False,
    # ext_modules=ext,
    entry_points="""
    [console_scripts]
    plot_wing=openaerostruct.utils.plot_wing:disp_plot
    plot_wingbox=openaerostruct.utils.plot_wingbox:disp_plot
    """,
)
