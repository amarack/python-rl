
DIRS = basis/CTiles environments/libPOMDP

all:
	-for d in $(DIRS); do (cd $$d; cmake .; $(MAKE) ); done

clean:
	-for d in $(DIRS); do (cd $$d; cmake .; $(MAKE) clean ); done
	-find . -type f -name "*.pyc" -exec rm -f {} \;