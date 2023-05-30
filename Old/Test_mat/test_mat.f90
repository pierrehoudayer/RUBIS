       program test_mat

       implicit none
       integer, parameter :: kl = 129, ku = 129, n = 106184, ldmat = 2*kl+ku+1
       double precision :: mat(ldmat,n), work(3*n), rr(n), cc(n)
       double precision :: anorm1, anorm_inf, rcond1, rcond_inf, rowcnd, colcnd
       double precision :: amax
       double precision, external :: DLANGB
       integer :: i, ii, j, iwork(n), info, ipiv(n)

       open(unit=31,file="ab.txt",status="old")
       read(31,*) ! skip header
       do i=1,ldmat
         read(31,*) (mat(i,j),j=1,n)
       enddo
       close(31)

       ! calculate 1 and infinity norms of mat
       call DGBEQU(n,n,kl,ku+kl,mat,ldmat,rr,cc,rowcnd,colcnd,amax,info)
       print*,"DGBEQU: ",info,rowcnd,colcnd,amax
       do j=1,n
         do ii=kl+1, 2*kl+ku+1
           i = ii+j-kl-ku-1
           if (i.lt.1) cycle
           if (i.gt.n) cycle
           mat(ii,j) = mat(ii,j)*rr(i)*cc(j)
         enddo
       enddo

       !open(unit=31,file="ab_equ.txt",status="unknown")
       !do i=1,ldmat
       !  write(31,*) (mat(i,j),j=1,n)
       !enddo
       !close(31)

       !open(unit=31,file="r_and_c.txt",status="unknown")
       !do i=1,n
       !  write(31,*) rr(i), cc(i)
       !enddo
       !close(31)

      
       anorm1 = DLANGB("1",n,kl,ku,mat,ldmat,work)
       anorm_inf = DLANGB("I",n,kl,ku,mat,ldmat,work)

       call DGBTRF(n,n,kl,ku,mat,ldmat,ipiv,info)

       call DGBCON("1",n,kl,ku,mat,ldmat,ipiv,anorm1, &
                   rcond1,work, iwork, info)
       call DGBCON("I",n,kl,ku,mat,ldmat,ipiv,anorm_inf, &
                   rcond_inf,work, iwork, info)
       print*,"Condition number (one-norm): ", rcond1
       print*,"Condition number (inf-norm): ", rcond_inf

       end program test_mat
