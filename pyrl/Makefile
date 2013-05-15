
DIRS = basis/CTiles/build environments/libPOMDP/build environments/mdptetris/build

all:
	-for d in $(DIRS); do (mkdir $$d; cd $$d; cmake ..; $(MAKE) ); done

clean:
	-for d in $(DIRS); do (mkdir $$d; cd $$d; cmake ..; $(MAKE) clean; cd ..; rm -rf build); done
	-find . -type f -name "*.pyc" -exec rm -f {} \;