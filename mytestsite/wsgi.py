"""
WSGI config for mytestsite project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application

sys.path.append(r"C:\django\SER")
##sys.path.append(r"C:\Users\nlplab\.conda\envs\yxzeng-demo\Lib\site-packages")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mytestsite.settings')

application = get_wsgi_application()
