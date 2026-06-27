def test_core_modules_import():
    import helpers
    import rubis.legendre
    import rubis.numerical
    import polytrope
    import rotation_profiles
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "1.1.0"