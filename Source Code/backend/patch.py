import eventlet
eventlet.monkey_patch()
print("Eventlet monkey patching applied from patch.py")

# Expose eventlet for use in other modules
eventlet_module = eventlet