# HUSTEF 2021 - Robot framework parallel runner PoC


This repository contains the proof of concept code necessary to reproduce the demo shown in the talk *Beat time - Exploiting parallelism in test execution* by Attila Tóth presented at testing conference HUSTEF 2021. The code shows how Robot framework test cases can be grouped automatically into clusters and then executed in parallel on multiple worker processes.

## Quickstart

1. Create a virtualenv and install dependencies:

   Linux:

    ```bash
   virtualenv -p `which python3` venv
   source venv/bin/activate
   pip3 install -r requirements.txt
    ```

    Windows:

    ```
   virtualenv.exe venv
   venv\Scripts\activate.bat
   pip3 install -r requirements.txt
    ```

2. Run the python script:

    ```bash
    python3 poc_parallel_runner.py --suites-dir ./suites --parallel-workers 4
    ```

## Documentation

The runner script (`poc_parallel_runner.py`) will parse all suites from the given directory, group them into clusters, create and execute the cluster configuration test case and then execute test cases within that cluster in parallel. Test cases in the same cluster are assumed to be compatible with each other. The clusters are determined based on the configuration needs of the test cases. Each test case might have a tag in the form of  "config-" and the name of the configuration parameter equaling the value of that resource or configuration ("config-<parameter name>=<value>"). If a given parameter is not defined for a test case, then it means that the test case does not care about that configuration parameter and can run with any value set on the system under test.

Cluster configuration test cases are created from `config_suite/config_suite.robot` where each configuration parameter should have a setter keyword e.g. `Set param1` for `config-param1`. These parameter names should also be defined in the `poc_parallel_runner.py` script within the `class ConfigData`

The suites are parsed by default from `./suites` which can be overridden with `--suites-dir`

Number of worker processes is defined by `--parallel-workers`. It can be set to any reasonable number. We tested it with 4 and 6 workers, but it should work with higher numbers as well.

For demonstration purpuses, there is a suite in `suites/example_with_config_tag.robot` and a matching `config_suite/config_suite.robot`. These are executed by `poc_parallel_runner.py.` If you want to experiment with the runner logic, try modifying these robot files first.

Please check the conference talk for more details about the logic and implementation!

## Contribution

This code is provided for demonstration purposes, therefore no contribution is expected to this repository. However, please feel free to fork and use it as a basis for further customization.

## Credits

This project would not be possible without the [Robot Framework](https://robotframework.org) and without the work of several enthusiastic people at Nokia, who contributed to the implementation of the runner. Among others, many thanks to Attila Jány and Bertalan Pécsi for debugging and refining the runner!

## License

This repository is licensed under the terms of Apache License v2.0. Please check [LICENSE](LICENSE) for details.

© Nokia 2021