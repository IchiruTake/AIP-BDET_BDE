# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
import string
import secrets
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from web.backend.backend import RunPrediction
from web.config.config import OnStartup
from web.backend.validation import GetCanonicalSmiles

# --------------------------------------------------------------------------------

FASTAPI_CONFIG = {
    'DEBUG': False,
    'TITLE': 'AIP-BDET',
    'SUMMARY': 'AIP-BDET: Bond Dissociation Energy Prediction with Layers of Isolated Substructure '
               'in Organic Molecules under the Lightweight Deep Feed-Forward Networks',
    'DESCRIPTION': 'AIP-BDET: An Accurate and Instant Prediction from Bond Dissociation Energy Tool '
                   'that can predict the BDE at high accuracy with minimal cost.',
    'VERSION': '0.1.0',
    'OPENAPI_URL': '/openapi.json',
    'DOCS_URL': '/docs',
    'REDOC_URL': '/redoc',
    'SWAGGER_UI_OAUTH2_REDIRECT_URL': '/docs/oauth2-redirect',
    # https://fastapi.tiangolo.com/tutorial/middleware/
    'MIDDLEWARE': [Middleware(SessionMiddleware, secret_key='3D278F7C8DC35')],
    'ON_STARTUP': (OnStartup,),
    'ON_SHUTDOWN': None,
    'LIFESPAN': None,
    'TERMS_OF_SERVICE': None,
    'CONTACT': {
        'name': 'Ichiru Take',
        'url': 'https://github.com/IchiruTake',
        'email': 'P.Ichiru.HoangMinh@gmail.com',
    },
    'LICENSE_INFO': {
        # MIT License
        'name': 'MIT License',
        'url': 'https://opensource.org/license/mit/',
    },
    'ROOT_PATH': '/',
    'ROOT_PATH_IN_SERVERS': True,
    'RESPONSES': None,
    'CALLBACKS': None,
    'WEBHOOKS': None,
    'DEPRECATED': None,
    'SEPARATE_INPUT_OUTPUT_SCHEMAS': True,
    'INCLUDE_IN_SCHEMA': True,
}

app: FastAPI = FastAPI(
    debug=FASTAPI_CONFIG.get('DEBUG', False),
    title=FASTAPI_CONFIG.get('TITLE'),
    summary=FASTAPI_CONFIG.get('SUMMARY'),
    description=FASTAPI_CONFIG.get('DESCRIPTION'),
    version=FASTAPI_CONFIG.get('VERSION'),
    openapi_url=FASTAPI_CONFIG.get('OPENAPI_URL'),
    docs_url=FASTAPI_CONFIG.get('DOCS_URL'),
    redoc_url=FASTAPI_CONFIG.get('REDOC_URL'),
    swagger_ui_oauth2_redirect_url=FASTAPI_CONFIG.get('SWAGGER_UI_OAUTH2_REDIRECT_URL'),
    middleware=FASTAPI_CONFIG.get('MIDDLEWARE', None),
    on_startup=FASTAPI_CONFIG.get('ON_STARTUP', None),
    on_shutdown=FASTAPI_CONFIG.get('ON_SHUTDOWN', None),
    lifespan=FASTAPI_CONFIG.get('LIFESPAN', None),
    terms_of_service=FASTAPI_CONFIG.get('TERMS_OF_SERVICE'),
    contact=FASTAPI_CONFIG.get('CONTACT'),
    license_info=FASTAPI_CONFIG.get('LICENSE_INFO', None),
    root_path=FASTAPI_CONFIG.get('ROOT_PATH', '/'),
    root_path_in_servers=FASTAPI_CONFIG.get('ROOT_PATH_IN_SERVERS', True),
    responses=FASTAPI_CONFIG.get('RESPONSES', None),
    callbacks=FASTAPI_CONFIG.get('CALLBACKS', None),
    webhooks=FASTAPI_CONFIG.get('WEBHOOKS', None),
    deprecated=FASTAPI_CONFIG.get('DEPRECATED', None),
    separate_input_output_schemas=FASTAPI_CONFIG.get('SEPARATE_INPUT_OUTPUT_SCHEMAS', True),
    include_in_schema=FASTAPI_CONFIG.get('INCLUDE_IN_SCHEMA', True),
)

# [1]: Load website
# Flash message: https://medium.com/@arunksoman5678/fastapi-flash-message-like-flask-f0970605031a
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

REPORT = {}
def update_message(response: dict):
    for key, value in response.items():
        REPORT[key] = value
    return REPORT


# --------------------------------------------------------------------------------
# Application Function
@app.get(path='/', response_class=HTMLResponse)
async def index(request: Request):
    # Redirect to /index.html
    response = {'request': request}
    return templates.TemplateResponse('index.html', context=response)


@app.post(path='/prediction', status_code=status.HTTP_201_CREATED, response_class=HTMLResponse)
async def prediction(request: Request, smiles: str = Form(), to_csv: bool = Form(False)):
    global REPORT
    print('Received request with smiles =', smiles)
    REPORT.clear()
    response = {'request': request}

    # Preprocess the SMILES
    processed_smiles_list = []
    smiles_list: list = smiles.split(',')
    for i, s in enumerate(smiles_list):
        c_smiles: Optional[str] = GetCanonicalSmiles(smiles=s.strip(), mode='SMILES', ignore_error=True)
        if c_smiles is not None:
            processed_smiles_list.append(c_smiles)

    if len(processed_smiles_list) == 0:
        redirect_url = request.url_for('index')
        update_message(response)
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    docs = []
    for i, s in enumerate(processed_smiles_list):
        doc = {}
        report = RunPrediction(smiles=s, mode='SMILES')
        doc['smiles'] = s
        doc['df'] = report['df']
        doc['report'] = report['report']
        doc['is_trained'] = report['is_trained']
        doc['has_untrained_atoms'] = len(report['atomic_state']['non_train']) > 0
        docs.append(doc)

    if len(smiles_list) == 1:
        response['smiles'] = processed_smiles_list[0]
        response['df'] = docs[0]['df']
        response['report'] = docs[0]['report']
        response['is_trained'] = docs[0]['is_trained']
        response['has_untrained_atoms'] = docs[0]['has_untrained_atoms']
        if not to_csv:
            redirect_url = request.url_for('output')
            update_message(response)
            return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    if to_csv:
        string_s = string.ascii_letters + string.digits
        FILENAME = 'temp_pred_' + ''.join(secrets.choice(string_s) for _ in range(6)) + '.csv'
        if len(smiles_list) == 1:
            docs[0]['df'].to_csv(FILENAME, index=False)
        else:
            pd.concat([d['df'] for d in docs], axis=0).to_csv(FILENAME, index=False)
        return FileResponse(path=FILENAME, media_type="text/csv", filename=FILENAME)

    response['docs'] = docs
    redirect_url = request.url_for('multi_output')
    update_message(response)
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@app.get('/result', response_class=HTMLResponse)
async def output(request: Request):
    REPORT['request'] = request
    return templates.TemplateResponse('result.html', context=REPORT)

@app.get('/multiresult', response_class=HTMLResponse)
async def multi_output(request: Request):
    REPORT['request'] = request
    return templates.TemplateResponse('multi_result.html', context=REPORT)
