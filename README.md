# Workflow for Automated Design of Hybrid Halide Perovskite Monolayers

## Usage

#### Create a virtual environment using micromamba

```shell
micromamba create -n aiida -c conda-forge aiida ase pymatgen cclib mendeleev spglib numpy scipy
```

#### Activate the virtual environment

```shell
conda activate aiida
```

#### Install aiida plugins

```shell
pip install aiida-nwchem aiida-siesta aiida-vasp aiida-shell aiida-pseudo
```

#### Initial postgresql database
```shell
initdb -D <path_to_postgresql_data_directory>
pg_ctl -D <path_to_postgresql_data_directory> -l logfile start
```

#### Create postgresql database
```shell
psql -d postgres
```
```postgresql
create database "HHPM";
```
```postgresql
\q
```

#### Setup AiiDA profile using created HHMP database
```shell
verdi setup
```

#### Start RabbitMQ server
```shell
rabbitmq-server -detached
```

#### Start AiiDA daemon
```shell
verdi daemon start 4
```

#### Check AiiDA status
```shell
verdi status
```

#### Clone this repository, and register the workflow

```shell
git clone 
cd workflow_hhpm
pip install -e .
```

#### Config pseudopotentials

```shell
aiida-pseudo install pseudo-dojo -f psml
verdi data vasp-potcar uploadfamily --path=<path_to_VASP_PAW_PBE_potential_folder> --name="PAW_PBE" --description="PAW_PBE"
```

#### Create computer

create_computer.yaml

```yaml
label: "<computer_label>"
hostname: "<computer_address>"
description: "<computer_description>"
transport: "core.ssh"
scheduler: "core.slurm"
shebang: "#!/bin/bash"
work_dir: "<directory_for_aiida_calculations>"
mpirun_command: "mpirun -np {tot_num_mpiprocs}"
mpiprocs_per_machine: <n_core_per_node>
default_memory_per_machine: <memory_in_kb_per_node>
use_double_quotes: true
prepend_text: ~
append_text: ~
```

```shell
verdi computer setup -n --config create_computer.yaml
```

config_computer.yaml

```yaml
user: "<aiida_user>"
username: "<host_user>"
port: <port>
look_for_keys: false
key_filename: "<path_to_private_key>"
timeout: 600
allow_agent: false
proxy_jump: ~
proxy_command: ~
compress: true
gss_auth: false
gss_kex: false
gss_deleg_creds: false
gss_host: "<hostname>"
load_system_host_keys: true
key_policy: "AutoAddPolicy"
use_login_shell: false
safe_interval: 30
```

```shell
verdi computer configure core.ssh <computer_label> -n --config config_computer.yaml
```

#### Create code

nwchem_code.yaml

```yaml
computer: "<computer_label>"
filepath_executable: "<path_to_nwchem_executable>"
label: "nwchem"
description: "nwchem@<computer_label>"
default_calc_job_plugin: "nwchem.nwchem"
use_double_quotes: true
with_mpi: true
prepend_text: |
  eval "$(micromamba shell hook --shell bash)" >/dev/null
  micromamba activate nwchem
  export OMP_NUM_THREADS=1
append_text: ~
```

siesta_code.yaml

```yaml
computer: "<computer_label>"
filepath_executable: "<path_to_siesta_executable>"
label: "siesta"
description: "siesta@<computer_label>"
default_calc_job_plugin: "siesta.siesta"
use_double_quotes: true
with_mpi: true
prepend_text: |
  eval "$(micromamba shell hook --shell bash)" >/dev/null
  micromamba activate siesta
  export OMP_NUM_THREADS=1
append_text: ~
```

vasp_code.yaml

```yaml
computer: "<computer_label>"
filepath_executable: "<path_to_vasp_executable>"
label: "vasp"
description: "vasp@<computer_label>"
default_calc_job_plugin: "vasp.vasp"
use_double_quotes: true
with_mpi: true
prepend_text: |
  module load vasp.6.4.0
append_text: ~
```

syva_code.yaml

```yaml
computer: "<computer_label>"
filepath_executable: "<path_to_syva_executable>"
label: "syva"
description: "syva@<computer_label>"
default_calc_job_plugin: "syva.syva"
use_double_quotes: true
with_mpi: false
prepend_text: ~
append_text: ~
```

Create `code`

```shell
verdi code create core.code.installed -n --config nwchem_code.yaml
verdi code create core.code.installed -n --config siesta_code.yaml
verdi code create core.code.installed -n --config vasp_code.yaml
verdi code create core.code.installed -n --config syva_code.yaml
```

#### Write submit script

submit.py

```python
from aiida import orm, load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from ase.io import read

load_profile("<profile_name>")

inputs = {
    "nwchem": {
        "code": orm.load_code("nwchem@<computer_label>"),
        "metadata": {
            "options": {
                "max_wallclock_seconds": 172800,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 64,
                },
                "queue_name": "normal",
                "withmpi": True,
            },
        },
    },
    "syva": {
        "code": orm.load_code("syva@<computer_label>"),
        "metadata": {
            "options": {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 64,
                },
                "queue_name": "normal",
                "parser_name": "hhpm.syva",
                "withmpi": False,
            },
        },
    },
    "siesta": {
        "code": orm.load_code("siesta@<computer_label>"),
        "metadata": {
            "options": {
                "max_wallclock_seconds": 172800,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 64,
                },
                "queue_name": "normal",
                "withmpi": True,
            },
        },
    },
    "vasp": {
        "code": orm.load_code("vasp@<computer_label>"),
        "metadata": {
            "options": {
                "max_wallclock_seconds": 172800,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 64,
                },
                "queue_name": "normal",
                "withmpi": True,
            },
        },
        "ncore": 1,
    },
    "shell": {
        "metadata": {
            "options": {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 64,
                },
                "queue_name": "normal",
                "withmpi": False,
            },
        },
    },
    "A": orm.StructureData(ase=read("<path_to_cation_structure>")),
    "B": orm.Str("<metal_symbol>"),
    "X": orm.Str("<halogen_symbol>"),
}

submit(WorkflowFactory("hhpm.main"), **inputs)
```

#### Submit workflow

```shell
verdi run submit.py
```
