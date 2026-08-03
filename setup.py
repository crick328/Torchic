from setuptools import setup, find_packages
from setuptools.command.install import install
import sys

'''
Uso:
Per installare solo la versione CPU: pip install .
Per poter usare la GPU: pip install .[cuda] (richiede software esterno)
Per avere le funzionalità grafico/visive: pip install .[visual] (richiede software esterno)
Per installare la versione completa: pip install .[full]

Nota: Per utilizzare CuPy, assicurati di avere installato CUDA da: https://developer.nvidia.com/cuda-downloads.
Nota: Per utilizzare le funzionalità visuali, installa l'engine Graphviz: https://pypi.org/project/graphviz/
'''




class PostInstallCommand(install):
    """Custom post-installation for user messages."""
    def run(self):
        install.run(self)
        extras = list(self.distribution.extras_require.keys())

        with open('setup_args.log', 'w') as f:
            for extra in extras:
                f.write(extra)

        if 'cupy' in extras:
            sys.stderr.write("Nota: CuPy richiede le librerie CUDA. Assicurati di averle installate correttamente dal sito Nvidia: https://developer.nvidia.com/cuda-downloads")
        if 'visual' in extras:
            sys.stderr.write("\n\nNota: Graphviz richiede l'installazione manuale dell'engine. Per maggiori informazioni: https://pypi.org/project/graphviz/")


# Base requirements (common to all installations)
base_requirements = [
    'numpy>=1.26.2',
    'typing_extensions>=4.9.0',
    'safetensors>=0.4.3',
    'datasets',
    # Semi-optionals
    'kaggle',
    'pandas'    # used only for SG Dataset
]

# GPU (CuPy) requirements
gpu_requirements = [
    'cupy-cuda12x>=14.0.0',
    'ml-dtypes'     # for bfloat16
]

# Visual (matplotlib and graphviz) requirements
visual_requirements = [
    'matplotlib',
    'graphviz==0.20.3'
]

# Extra packages for optional installations
extras = {
    'cuda': gpu_requirements,
    'visual': visual_requirements,
    'full': base_requirements + gpu_requirements + visual_requirements,
}

setup(
    name='Torchic',
    version='3.2',
    packages=find_packages(),
    install_requires=base_requirements,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'torchic-cli=torchic.cli:main',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        #'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)