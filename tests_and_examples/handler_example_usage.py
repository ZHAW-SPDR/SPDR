from spdr.handler import SPDR_RT09_Handler


if __name__ == '__main__':
    handler = SPDR_RT09_Handler()
    element = handler.run()
    handler.do_cleanup()