import math
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time}</green> <magenta>{level}</magenta> {message}", level="WARNING")
logger.add("my_log.log", level="DEBUG", rotation="100 MB")

logger.debug("You have entered the debug zone!")
logger.info("You have entered the info void!")
logger.warning("You have entered the warning realm!")
logger.error("You have entered the error domain!")
logger.critical("You have entered the critical region!")

logger.opt(lazy=True).debug("If sink <= DEBUG: {x}", x=lambda: math.factorial(2**5))


@logger.catch
def dangerous_function():
    1 / 0


dangerous_function()
