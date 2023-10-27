import io, os, time, signal
from threading import Thread
from flask import Flask, request, jsonify, Response
import logging
import asyncio
import sys
import argparse

sys.path.append("..")
from controller.llm_controller import LLMController

logging.disable(logging.CRITICAL + 1)
app = Flask(__name__)
FRAME_RATE = 30

global llm_controller
main_page = open('./index.html', 'r').read()

def shutdown():
    print("Shutting down gracefully...")
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)

def process_command(command: str):
    print(f"Received command: {command}")
    if command == "exit":
        llm_controller.stop_controller()
        Thread(target=shutdown).start()
    else:
        llm_controller.execute_user_command(command)

def generate_mjpeg_stream():
    while True:
        image_data = llm_controller.get_latest_frame()
        buf = io.BytesIO()
        image_data.save(buf, format='JPEG')
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(1.0 / FRAME_RATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return main_page
    
@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    process_command(user_input)
    return jsonify({'result': 'success'})

# asyncio functions
global asyncio_loop
asyncio_loop = asyncio.get_event_loop()
def start_async_loop():
    global asyncio_loop
    asyncio_loop.run_forever()

async def stop_async_loop():
    global asyncio_loop
    asyncio_loop.stop()

def stop_loop_from_thread():
    global asyncio_loop
    # Schedule the stopping function to run on the loop
    asyncio_loop.call_soon_threadsafe(asyncio_loop.create_task, stop_async_loop())

def init_llm_controller(v=False):
    global llm_controller
    print("init llm controller")
    llm_controller = LLMController(v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--virtual-camera", action='store_true', help="Use laptop camera")
    args = parser.parse_args()
    init_llm_controller(args.virtual_camera)
    # Start the asyncio loop in a separate thread
    async_thread = Thread(target=start_async_loop)
    async_thread.start()

    # Start the LLM controller
    llm_controller.start_robot()
    
    # Start the LLM capture loop in a separate thread
    llm_thread = Thread(target=llm_controller.capture_loop, args=(asyncio_loop,))
    llm_thread.start()

    # Start the Flask server
    app.run(host='localhost', port=50001, debug=True, use_reloader=False)

    # Stop the LLM controller
    stop_loop_from_thread()
    llm_thread.join()
    async_thread.join()
    llm_controller.stop_robot()