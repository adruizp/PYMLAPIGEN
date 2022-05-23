import pymlapigen
import os


if (__name__ == "__main__"):
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.join(os.getcwd(), "pymlapigen")
    pymlapigen.flask_app.run(debug=True)