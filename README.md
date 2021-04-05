# Environment setup

To facilitate the management of dependencies it is strongly advisable to create a virtual environment from the ```environment.yml``` file in this directory.

From the project directory, I suggest doing it with ```conda``` following these steps:

1. It is useful to create an environment that lives in the working directory rather than under ```/anaconda3/envs```. To avoid some minor drawbacks of this approach, modify the ```env_prompt``` setting in your ```.condarc``` file by typing the following in the terminal:

      ```
      conda config --set env_prompt '({name}) '
      ```
      Then create the environment:

      ```
      conda env create --prefix ./hep-lfi --file ./environment.yml
      ```

2. To activate it, use the following (again in the project directory):

      ```
      conda activate ./hep-lfi
      ```
      To deactivate the environment (will go back to base env):

      ```
      conda deactivate
      ```
