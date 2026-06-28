def test_core_modules_import():
    import helpers
    import rubis.legendre
    import rubis.numerical
    import rubis.polytrope as polytrope
    import rubis.rotation_profiles as rotation_profiles
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "1.1.0"
    
def test_helpers_reexports_find_domains():
    from helpers import find_domains as legacy_find_domains
    from rubis.domains import find_domains

    assert legacy_find_domains is find_domains