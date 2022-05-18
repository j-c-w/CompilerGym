#!/bin/env python3
from CGRA import CgraEnv, compilation_session_cgra
import compiler_gym.service.runtime as runtime
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    print ("Starting execution of compiler_gym ")
    logging.info("Staring execugion")
    logging.info("For CGRA" + str(compilation_session_cgra))
    runtime.create_and_run_compiler_gym_service(CGRACompilationSession)
    logging.info("Finished Execution")