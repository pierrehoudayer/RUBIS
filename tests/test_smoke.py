def test_core_modules_import():
    import rubis._utils as utils
    import rubis.domains as domains
    import rubis.legendre as legendre
    import rubis.mapping as mapping
    import rubis.numerical as numerical
    import rubis.polytrope as polytrope
    import rubis.rotation_profiles as rotation_profiles
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "1.1.0"