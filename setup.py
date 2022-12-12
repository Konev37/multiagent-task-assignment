from setuptools import setup, find_packages

setup(name='mata',
      version='0.0.1',
      description='Multi-Agent Task Assignment Environment',
      url='https://github.com/openai/multiagent-public',
      author='Bagration C',
      author_email='bagration_c@163.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
