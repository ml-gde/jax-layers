#!/usr/bin/env python
"""Script to run all tests for the jax-layers project."""

import os
import sys

import pytest


def main():
    """Run all tests for the jax-layers project."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the root directory of the project
    root_dir = os.path.dirname(script_dir)

    # Add the root directory to the Python path
    sys.path.insert(0, root_dir)

    # Run the tests
    args = [
        "-xvs",  # -x: exit on first failure, -v: verbose, -s: don't capture stdout
        script_dir,  # Run all tests in the tests directory
        "--cov=jax_layers",  # Generate coverage report for jax_layers
        "--cov-report=term",  # Output coverage report to terminal
    ]

    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])

    # Run pytest with the arguments
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
