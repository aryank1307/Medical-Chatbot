from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from fastapi.responses import JSONResponse
try:
    from server.modules.load_vectorstore import load_vectorstore
    from server.logger import logger
except ModuleNotFoundError:
    from modules.load_vectorstore import load_vectorstore
    from logger import logger


router=APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    specialty: str = Form("general"),
    namespace: str = Form("medical-guidelines"),
):
    try:
        logger.info("Recieved uploaded files")
        load_vectorstore(files, specialty=specialty, namespace=namespace)
        logger.info("Document added to vectorstore")
        return {
            "messages": "Files processed and vectorstore updated",
            "specialty": specialty,
            "namespace": namespace,
        }
    except RuntimeError as e:
        logger.warning(str(e))
        return JSONResponse(status_code=400,content={"error":str(e)})
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500,content={"error":str(e)})
