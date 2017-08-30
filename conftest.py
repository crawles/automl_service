def pytest_addoption(parser):
    parser.addoption("--host", action="store",
        help="run all combinations")

def pytest_generate_tests(metafunc):
    if 'host' in metafunc.fixturenames:
        option_value = metafunc.config.option.host
        print option_value
        if option_value:
            metafunc.parametrize("host", [option_value])
