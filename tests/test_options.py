from rubis.options import (
    OutputOptions,
    DiagnosticOptions,
    PlotOptions,
    RadiativeFluxOptions,
    ModelOutputOptions,
)

def test_output_options_builds_all_suboptions():
    options = OutputOptions()

    assert isinstance(options.diagnostics, DiagnosticOptions)
    assert isinstance(options.plot, PlotOptions)
    assert isinstance(options.flux, RadiativeFluxOptions)
    assert isinstance(options.model, ModelOutputOptions)