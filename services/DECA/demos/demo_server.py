import base64
import hashlib
import os, sys

_deca_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_deca_root)
sys.path.insert(0, _deca_root)

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from demos.demo_reconstruct import output_from_image, init_my_deca

server = FastAPI()

deca = init_my_deca()


def img_to_hash(img: np.ndarray) -> str:
    return hashlib.sha256(img).hexdigest()


@server.post('/')
async def serve_process_image(request: Request):
    content = await request.json()

    deca.cfg.model.extract_tex = content['include_tex']

    img_buffer = base64.b64decode(content['numpy_img'])
    img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(content['numpy_shape'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if h > w:
        img = img[(h - w) // 2:(h + w) // 2, :, :]
    else:
        img = img[:, (w - h) // 2:(w + h) // 2, :]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    img_hash = img_to_hash(img)
    cv2.imwrite(f'temp/{img_hash}.png', img)

    with_pose = content.get('with_pose', False)
    data = output_from_image(img, deca, no_detect_pose=not with_pose)

    return Response(content=data, media_type='application/zip', headers={'Image-Hash': img_hash})


@server.post('/update')
async def update_face_parameters(request: Request):
    content = await request.json()

    deca.cfg.model.extract_tex = content['includeTex']

    img = cv2.imread(f"temp/{content['imgHash']}.png")

    data = output_from_image(
        img, deca,
        emotion_arr=content.get('emotionArr'),
        exp_arr=content.get('expArr'),
        pose_arr=content.get('poseArr'),
        neck_pose_arr=content.get('neckPoseArr'),
        eye_pose_arr=content.get('eyePoseArr'),
    )

    return Response(content=data, media_type='application/zip')


if __name__ == '__main__':
    uvicorn.run('demos.demo_server:server', host='0.0.0.0', port=11200, reload=True)
