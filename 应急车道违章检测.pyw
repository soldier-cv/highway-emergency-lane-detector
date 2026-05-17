"""双击启动应急车道违章检测GUI（无控制台窗口）"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency_lane"))
from traffic_violation_gui import main

main()
