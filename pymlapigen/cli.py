import pymlapigen
import os

def cli():
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.dirname(pymlapigen.__file__)
    pymlapigen.flask_app.run()