def test_core_modules_import():
    import helpers
    import legendre
    import numerical
    import polytrope
    import rotation_profiles
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "1.1.0"