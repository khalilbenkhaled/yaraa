import sys
import subprocess
from yaraa.utils.parse_yaml import parse_yaml


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_yaml>")
        sys.exit(1)
    yaml_file = sys.argv[1]
    r, inference, evaluator = parse_yaml(yaml_file)

    if inference:
        # Start the frontend server
        if inference['frontend'] == 'streamlit':
            frontend_command = 'streamlit run ui.py'
            process = subprocess.Popen(frontend_command, shell=True)

        # Start the backend server
        if inference['backend'] == 'fastapi':
            import uvicorn
            from fastapi import FastAPI
            #### API CODE ####
            app = FastAPI()

            @app.get("/chat/")
            async def echo_string(question: str):
                result = r.generate(question)
                return {"reply": result[0]}
            uvicorn.run(app, host="0.0.0.0", port=8000)
    elif evaluator:
        questions = evaluator['synthetic_data_generator'].generate()
        scores = evaluator['evaluator'].evaluate(questions)


if __name__ == '__main__':
    main()
