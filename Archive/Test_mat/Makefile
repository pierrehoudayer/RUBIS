F77      = gfortran
F90      = gfortran
FFLAGS   = -O2 -fbounds-check #-fcheck=all 
BLAS     = -lopenblas -lpthread
LAPACK   = -llapack
PROGRAM  = test_mat
SRCS90   = test_mat.f90
OBJS90   = $(SRCS90:.f90=.o)
OBJS     = $(SRCS:.f=.o)

$(PROGRAM): $(OBJS) $(OBJS90)
	$(F90) $(FFLAGS) $(OBJS) $(OBJS90) $(LAPACK) $(BLAS) -o $(PROGRAM)
	size $(PROGRAM)

$(OBJS90):%.o:%.f90
	@echo Compile $<
	@$(F90) $(FFLAGS) -c $<

$(OBJS):%.o:%.f
	@echo Compile $<
	@$(F77) $(FFLAGS) -c $<

.PHONY: clean rmall

clean:
	$(RM) $(OBJS) $(OBJS90) *.mod

rmall: clean
	$(RM) $(PROGRAM)
	(cd $(ROOTDIR)/lib && $(MAKE) $@)

