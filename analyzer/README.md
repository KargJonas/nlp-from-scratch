# ML Analyzer
### This is a tool for visualizing training performance of custom language model.
This tool (currently) works by reading csv files with only one (unnamed) column.
All relevant configuration is done through environment variables. The defaults are defined in `.env`.

### How to start:
`$ docker-compose -f analyzer/docker-compose.yml up`

Alternatively, the metrics directory can be quickly set using something like this:

`$ METRICS_DIR=/my/custom/metrics/dir docker-compose -f analyzer/docker-compose.yml up`

There is also a `start.sh` script which starts the analyzer by running the first command.
