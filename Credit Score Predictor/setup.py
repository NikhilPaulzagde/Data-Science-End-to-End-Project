from setuptools import find_packages, setup
from typing import List

HPYEN_E_DOT = "-e ."

def get_requirements(filepath:str) -> List[str]:

    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]


        if HPYEN_E_DOT in requirements:
             requirements.remove(HPYEN_E_DOT)

        return requirements

   
setup(
      name='Credit Card Score Predictor',
      version='0.0.0.1',
      description='PipeLine Project FOr ML',
      author='Nikhil Paulzgade',
      author_email='nickpaulzagde@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
)
