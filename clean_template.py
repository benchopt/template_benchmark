import re
from pathlib import Path
from subprocess import check_output


if __name__ == "__main__":

    # Get the name of the repo and parse org and bench name
    repo_url = check_output(['git', 'remote', 'get-url', 'origin']).decode()
    name = re.split('github.com[:/]', repo_url)[1].strip()
    ORG, BENCHMARK_NAME = name.replace('.git', '').split('/')

    # update readme to have the right badge
    README = Path('README.rst')
    text = README.read_text()

    text = text.replace('#ORG', ORG)
    text = text.replace('#BENCHMARK_NAME', BENCHMARK_NAME)

    text = '\n'.join([line for line in text.splitlines()[13:]
                      if 'template_benchmark' not in line] + [''])

    README.write_text(text)
