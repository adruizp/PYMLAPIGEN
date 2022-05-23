""" Imports """
from flask import Flask

""" Aplicación Flask """
flask_app = Flask(__name__)

#Cargar configuracion predeterminada
flask_app.config.from_pyfile('config.py')

import pymlapigen.routes



