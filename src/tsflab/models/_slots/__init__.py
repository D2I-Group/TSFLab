"""Slot adapters: one uniform interface per recombination slot.

``registry`` is a torch-free table of slot options, their component dependencies,
and the compatibility rules; ``adapters`` holds the torch modules that wrap the
cataloged components exactly as each component card documents.
"""
