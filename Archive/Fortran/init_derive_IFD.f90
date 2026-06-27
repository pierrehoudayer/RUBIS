subroutine init_derive_IFD(derive,new_grid,grid,ngrid,der_max,order)
!------------------------------------------------------------------------------ 
!  This subroutine sets up a banded derivation matrices (which is stored in
!  general form) using nth order finite difference schemes.  The
!  derivatives are calculated with respect to an (almost) arbitrary grid.
!  The order of the scheme is 2*order, even though the number of points
!  in the window is also 2*order.  This can be achieved for 1st order
!  differential systems by carefully defining a new grid which is
!  approximately (or exactly, when order = 1) located at the grid mid-
!  points.  This method hopefully minimises the effects of mesh drift.
!  
!  The derivation coefficients are based on Lagrange interpolation
!  polynomials (cf. Fornberg, 1988).
!------------------------------------------------------------------------------ 
! Variables:
! 
! grid     = underlying on which the derivatives are calculated (grid(1:ngrid))
!            Warning: this grid must not contain two identical points.
! ngrid    = number of grid points (1:ngrid)
! order    = order of the scheme, divided by 2.
!------------------------------------------------------------------------------ 
! NOTE: the last line in the derive matrices is full of zeros (except
!       for derive(:,:,-1)).  This typically where one is expected to
!       place boundary conditions.  If one needs a boundary condition at
!       the first point of the domain, while keeping a banded structure
!       for the matrix, this can be achieved by setting up init_order
!       carefully.
!------------------------------------------------------------------------------ 
      implicit none
      integer, intent(in) :: ngrid, der_max, order
      double precision, intent(in) :: grid(ngrid)
      double precision, intent(out) :: derive(ngrid-1, ngrid, 0:der_max)
      double precision, intent(out) :: new_grid(ngrid-1)
      integer i, j, k, l, start, finish
      double precision prdct, num, den, xx, dx, acoef, bcoef
      double precision :: mu(-order:order), a(-1:der_max), aa(-1:2*order)
      double precision :: lbder(0:der_max), ubder(0:der_max)
      double precision, parameter :: eps = 1d-14
      integer, parameter :: n_iter_max = 100

      ! check parameters:
      if (der_max.lt.1) then
       stop "Please set der_max >= 1 in init_derive_IFD"
      endif
      if (order.lt.1) then
       stop "Please set order >= 1 in init_derive_IFD"
      endif
      if (order.lt.(der_max+1)/2) then
       stop "Please set order >= (der_max+1)/2 in init_derive_IFD"
      endif

      ! Initialisation
      do l=0,der_max
        ubder(l) = order
        lbder(l) = order-1
      enddo

      derive(:,:,:) = 0d0

      ! define new grid
      do i=1,ngrid-1

        start  = -order+1
        finish = order

        ! special treatment for the endpoints
        ! Note: this leads to results which are less precise on the
        ! edges, but preserves the banded form of the derivation matrix.
        if ((i+start).lt.1) start = 1 - i
        if ((i+finish).gt.ngrid) finish = ngrid - i

        ! scale polynomial
        acoef = 2d0/(grid(i+1)-grid(i))
        bcoef = (grid(i)+grid(i+1))/(grid(i)-grid(i+1))

        ! find polynomial coefficients:
        aa(:) = 0d0
        aa(0) = 1d0
        do j=start,finish
          do k=2*order,0,-1
            aa(k) = aa(k-1)-(acoef*grid(i+j)+bcoef)*aa(k)
          enddo
        enddo

        ! find a root of the polynomial: this will become the new grid
        ! point
        xx = 0d0
        dx = 1d0

        ! Newton type iteration
        j = 0
        do while ((abs(dx).gt.eps).and.(j.lt.n_iter_max))
          num = aa(2*order)*dble(2*order)
          do k=2*order-1,1,-1 
            num = xx*num+dble(k)*aa(k)
          enddo
          den = dble(2*order*(2*order-1))*aa(2*order)
          do k=2*order-1,2,-1 
            den = xx*den+dble(k*(k-1))*aa(k)
          enddo
          dx = num/den
          xx = xx - dx
          j  = j + 1
        enddo
        xx = (xx-bcoef)/acoef

        ! sanity check (just in case)
        if ((xx.le.grid(i)).or.(xx.ge.grid(i+1)))  then
          write(*,*) (grid(i+j),j=start,finish)
          stop "Error in init_derive_IFD: stray grid point"
        endif

        ! assign new grid point
        new_grid(i) = xx
      enddo


      do i=1,ngrid-1

        start  = -order+1
        finish = order

        ! special treatment for the endpoints
        ! Note: this leads to results which are less precise on the
        ! edges, but preserves the banded form of the derivation matrix.
        if ((i+start).lt.1) start = 1 - i
        if ((i+finish).gt.ngrid) finish = ngrid - i

        do j=start,finish
          mu(j) = grid(j+i) - new_grid(i)
        enddo

        do j=start,finish

          ! Initialise Lagrange polynomial
          a(-1) = 0d0
          a(0) = 1d0
          do l=1,der_max
            a(l) = 0d0
          enddo
          
          prdct = 1d0
          do k=start,j-1
            prdct = prdct*(mu(j)-mu(k))
          enddo
          do k=j+1,finish
            prdct = prdct*(mu(j)-mu(k))
          enddo
          a(0) = a(0)/prdct
          
          ! Calculate Lagrange polynomial, by calculating product of (x-mu(k))
          do k=start,j-1
            do l=der_max,0,-1
              a(l) = -mu(k)*a(l) + a(l-1)
            enddo
          enddo
          do k=j+1,finish
            do l=der_max,0,-1
              a(l) = -mu(k)*a(l) + a(l-1)
            enddo
          enddo

          ! The coeffecients a(l) of the Lagrange polynomial are the
          ! derivation coefficients divided by l! (where 0! = 1)
          prdct = 1d0
          do l=0,der_max
            prdct = prdct*dble(max(1,l))
            derive(i,i+j,l) = a(l)*prdct
          enddo
        enddo
      enddo

      return
end subroutine init_derive_IFD
