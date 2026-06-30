def test_core_modules_import():
    import rubis.domains as domains
    import rubis.legendre as legendre
    import rubis.mapping as mapping
    import rubis.models as models
    import rubis.numerical as numerical
    import rubis.options as options
    import rubis.poisson as poisson
    import rubis.polytrope as polytrope
    import rubis.results as results
    import rubis.rotation_profiles as rotation_profiles
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "1.1.0"