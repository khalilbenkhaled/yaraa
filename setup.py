from setuptools import setup, find_packages

setup(name='yaraa',
      version='0.0.1',
      description='test',
      author='khalilbenkhaled',
      license='MIT',
      packages=find_packages(),
      install_requires=['PyYAML', 'numpy', 'sentence-transformers'],
      extras_require={
          'transofmers': ['transformers'],
          'ollama': ['ollama'],
          'server': ['openai'],
          'chromadb': ['chromadb'],
          'streamlit': ['streamlit', 'requests'],
          'fastapi': ['fastapi', 'uvicorn'],
      }
)
