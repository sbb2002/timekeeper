# 디버깅을 위한 위치출력 처리
class PrintHandler:
    '''Location printer.
    This can print the message with instance's name.'''
    def prtwl(self, *text):
        '''Print-With-Location'''
        print(f"[{self.__class__.__name__}] ", *text)
