import pymlapigen
import sys
import os

def cli():
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.dirname(pymlapigen.__file__)
    if len(sys.argv) == 1:
        pymlapigen.flask_app.run()
    else:
        pymlapigen.flask_app.run(host=sys.argv[1])