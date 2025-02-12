from pprint import pprint

class Configuration:
    def __init__(self, **kwargs):
        # Define default parameters
        defaults = {
            'size': 100,
            'color': 'blue',
            'speed': 'fast',
            'enabled': True
        }
        
        # Update defaults with any provided values
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

def print_config(config, name):
    print(f"\n{name}:")
    print("-" * 40)
    for key, value in vars(config).items():
        print(f"{key}: {value}")

def run_tests():
    # Test 1: Create with all defaults
    config1 = Configuration()
    print_config(config1, "Test 1: All defaults")

    # Test 2: Override one parameter
    config2 = Configuration(color='red')
    print_config(config2, "Test 2: Override color only")

    # Test 3: Override multiple parameters
    config3 = Configuration(
        size=200,
        color='green',
        enabled=False
    )
    print_config(config3, "Test 3: Override multiple parameters")

    # Test 4: Override all parameters
    config4 = Configuration(
        size=300,
        color='yellow',
        speed='slow',
        enabled=False
    )
    print_config(config4, "Test 4: Override all parameters")

if __name__ == "__main__":
    run_tests()