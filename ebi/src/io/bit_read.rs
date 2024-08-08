use std::io::{self, Read};
/// Trait for reading bits from a byte stream.
pub trait BitRead {
    /// Read a single bit from the stream, and consume it.
    fn read_bit(&mut self) -> io::Result<bool>;

    /// Read a single byte from the stream, and consume it.
    fn read_byte(&mut self) -> io::Result<u8>;

    /// Read `n` bits from the stream, and consume them.
    /// If `n` is greater than 64, it will be handled as 64.
    /// If the stream's remaining bits are less than `n`, return an error(std::io::ErrorKind::UnexpectedEof).
    /// # Errors
    /// If the stream's remaining bits are less than `n`, an error([`std::io::ErrorKind::UnexpectedEof`]) will be returned.
    /// The stream will be consumed partially.
    fn read_bits(&mut self, n: u8) -> io::Result<u64>;

    /// Read `n` bits from the stream, and while not consuming them.
    /// If `n` is greater than 56, it will be handled as 56.
    /// Bits are stored in `u64` from the least significant bit.
    fn peak_bits(&mut self, n: u8) -> io::Result<u64>;
}

impl<T: BitRead> BitRead for &mut T {
    fn read_bit(&mut self) -> io::Result<bool> {
        (*self).read_bit()
    }

    fn read_byte(&mut self) -> io::Result<u8> {
        (*self).read_byte()
    }

    fn read_bits(&mut self, n: u8) -> io::Result<u64> {
        (*self).read_bits(n)
    }

    fn peak_bits(&mut self, n: u8) -> io::Result<u64> {
        (*self).peak_bits(n)
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct BitReader<R: Read> {
    /// The underlying reader.
    reader: R,
    /// The buffer of bits read from the reader.
    ///
    /// The buffer bits are stored in `buffer_pos`...`buffer_pos + buffer_len` bits.
    buffer: u64,
    /// The number of bits in the buffer. 0 indicates that the buffer is empty.
    buffer_len: u8,
}

impl<R: Read> std::fmt::Debug for BitReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitReader")
            .field("buffer", &format!("{:064b}", self.buffer))
            .field("buffer_len", &self.buffer_len)
            .finish()
    }
}

impl<R: Read> BitReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            buffer_len: 0,
        }
    }

    /// Fill the buffer with 64 bits from the reader.
    /// Return the number of bytes read.
    ///
    /// The bits left in the buffer will be preserved.
    #[inline]
    fn fill_buffer(&mut self) -> io::Result<u8> {
        if self.buffer_len == 64 {
            return Ok(0);
        }

        let rest_len = 64 - self.buffer_len;

        let n_bytes_fetchable = rest_len / 8;

        let mut bytes = [0u8; 8];

        let n_bytes_read = self.reader.read(&mut bytes[..n_bytes_fetchable as usize])? as u8;
        let n_bits_read = n_bytes_read * 8;

        let bytes = u64::from_be_bytes(bytes);

        // Create a mask to clear the buffer from `from` to `end`.
        // Offset is from the most significant bit.
        let create_mask = |from: u8, end: u8| {
            if end - from == 64 {
                return u64::MAX;
            }
            // let mask = (1 << (end - from)) - 1;
            // let mask = mask << (64 - end);
            1u64.wrapping_shl((end - from) as u32)
                .wrapping_sub(1)
                .wrapping_shl((64 - end) as u32)
        };
        let buffer_mask = create_mask(0, self.buffer_len);
        let bytes_read_mask = create_mask(self.buffer_len, self.buffer_len + n_bits_read);

        self.buffer =
            (self.buffer & buffer_mask) | ((bytes >> (self.buffer_len)) & bytes_read_mask);

        self.buffer_len += n_bits_read;

        Ok(n_bytes_read)
    }
}

impl<R: Read> BitRead for BitReader<R> {
    /// Read a single bit from the stream, and consume it.
    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use ebi::io::bit_read::{BitRead, BitReader};
    ///
    /// let mut reader = Cursor::new([0b1010_0101]);
    ///
    /// let mut bit_reader = BitReader::new(reader);
    ///
    /// assert_eq!(bit_reader.read_bit().unwrap(), true);
    /// assert_eq!(bit_reader.read_bit().unwrap(), false);
    /// assert_eq!(bit_reader.read_bit().unwrap(), true);
    /// assert_eq!(bit_reader.read_bit().unwrap(), false);
    ///
    /// assert_eq!(bit_reader.read_bit().unwrap(), false);
    /// assert_eq!(bit_reader.read_bit().unwrap(), true);
    /// assert_eq!(bit_reader.read_bit().unwrap(), false);
    /// assert_eq!(bit_reader.read_bit().unwrap(), true);
    /// ```
    #[inline]
    fn read_bit(&mut self) -> io::Result<bool> {
        if self.buffer_len >= 1 {
            let bit = (self.buffer >> 63) & 1;
            self.buffer <<= 1;
            self.buffer_len -= 1;

            return Ok(bit == 1);
        }
        self.fill_buffer()?;
        if self.buffer_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF while reading a bit",
            ));
        }

        let bit = (self.buffer >> 63) & 1;
        self.buffer <<= 1;
        self.buffer_len -= 1;

        Ok(bit == 1)
    }

    /// Read a single byte from the stream, and consume it.
    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use ebi::io::bit_read::{BitRead, BitReader};
    /// let mut reader = Cursor::new([0b1010_0101, 0b1111_0000]);
    ///
    /// let mut bit_reader = BitReader::new(reader);
    ///
    /// assert_eq!(bit_reader.read_byte().unwrap(), 0b1010_0101);
    /// assert_eq!(bit_reader.read_byte().unwrap(), 0b1111_0000);
    /// assert_eq!(bit_reader.read_byte().unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    /// ```
    #[inline]
    fn read_byte(&mut self) -> io::Result<u8> {
        if self.buffer_len >= 8 {
            let byte = (self.buffer >> 56) as u8;
            self.buffer <<= 8;
            self.buffer_len -= 8;

            return Ok(byte);
        }
        self.fill_buffer()?;

        if self.buffer_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF while reading a byte",
            ));
        }

        let byte = (self.buffer >> 56) as u8;
        self.buffer <<= 8;
        self.buffer_len -= 8;

        Ok(byte)
    }

    /// Read `n` bits from the stream, and consume them.
    /// If `n` is greater than 64, it will be handled as 64.
    ///
    /// # Errors
    /// If the stream's remaining bits are less than `n`, an error(io::ErrorKind::UnexpectedEof) will be returned.
    /// The stream will be consumed partially.
    ///
    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use ebi::io::bit_read::{BitRead, BitReader};
    ///
    /// let mut reader = Cursor::new([0b1010_0101, 0b1111_0000]);
    /// let mut bit_reader = BitReader::new(reader);
    ///
    /// assert_eq!(bit_reader.read_bits(1).unwrap(), 0b1);
    /// assert_eq!(bit_reader.read_bits(9).unwrap(), 0b010_0101_11);
    /// assert_eq!(bit_reader.read_bits(3).unwrap(), 0b110);
    /// assert_eq!(bit_reader.read_bits(4).unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    #[inline]
    fn read_bits(&mut self, mut n: u8) -> io::Result<u64> {
        // can't read more than 64 bits into a u64
        if n > 64 {
            n = 64;
        }

        if self.buffer_len > n {
            let bits = self.buffer >> (64 - n);
            self.buffer <<= n;
            self.buffer_len -= n;

            return Ok(bits);
        }

        let mut bits: u64 = 0;
        while n >= 8 {
            let byte = self.read_byte().map(u64::from)?;
            bits = bits.wrapping_shl(8) | byte;
            n -= 8;
        }

        while n > 0 {
            let bit = self.read_bit()?;
            bits = bits.wrapping_shl(1) | (bit as u64);

            n -= 1;
        }

        Ok(bits)
    }

    /// Read `n` bits from the stream, and while not consuming them.
    /// If `n` is greater than 56, it will be handled as 56.
    /// Bits are stored in `u64` from the most significant bit.
    ///
    /// # Example
    /// ```
    /// use std::io::Cursor;
    ///
    /// use ebi::io::bit_read::{BitRead, BitReader};
    ///
    /// let mut reader = Cursor::new([0b1010_0101, 0b1111_0000]);
    /// let mut bit_reader = BitReader::new(reader);
    ///
    /// assert_eq!(bit_reader.peak_bits(1).unwrap(), 0b1);
    /// bit_reader.read_bit();
    /// assert_eq!(bit_reader.peak_bits(4).unwrap(), 0b0100);
    /// ```
    #[inline]
    fn peak_bits(&mut self, mut n: u8) -> io::Result<u64> {
        if n > 56 {
            n = 56;
        }
        if self.buffer_len >= n {
            let bits = self.buffer >> (64 - n);
            return Ok(bits);
        }
        self.fill_buffer()?;

        if self.buffer_len < n {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF while peaking bits",
            ));
        }

        let bits = self.buffer >> (64 - n);

        Ok(bits)
    }
}

#[cfg(test)]
mod tests {

    use crate::io::bit_write::{BitWrite as _, BufferedBitWriter};

    use super::*;

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    #[allow(clippy::bool_assert_comparison)]
    fn test_bitreader() {
        let cursor = std::io::Cursor::new([0b1010_0101, 0b1111_0000, 0b1100_0011, 0xbe]);
        let mut bit_reader = BitReader::new(cursor);

        assert_eq!(bit_reader.read_bit().unwrap(), true);
        assert_eq!(bit_reader.peak_bits(3).unwrap(), 0b010);
        assert_eq!(bit_reader.read_bits(2).unwrap(), 0b01);

        assert_eq!(bit_reader.read_byte().unwrap(), 0b0_0101_111);

        assert_eq!(bit_reader.peak_bits(1).unwrap(), 0b1);
        assert_eq!(bit_reader.read_bits(5).unwrap(), 0b1_0000);

        assert_eq!(bit_reader.read_byte().unwrap(), 0b1100_0011);

        assert_eq!(bit_reader.read_byte().unwrap(), 0xbe);

        assert_eq!(
            bit_reader.read_bit().unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    #[allow(clippy::bool_assert_comparison)]
    fn read_bit() {
        let bytes = [0b01101100, 0b11101001];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), false);

        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), true);

        assert_eq!(
            b.read_bit().unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    #[allow(clippy::bool_assert_comparison)]
    fn read_byte() {
        let bytes = [100, 25, 0, 240, 240];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.read_byte().unwrap(), 100);
        assert_eq!(b.read_byte().unwrap(), 25);
        assert_eq!(b.read_byte().unwrap(), 0);

        // read some individual bits we can test `read_byte` when the position in the
        // byte we are currently reading is non-zero
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);
        assert_eq!(b.read_bit().unwrap(), true);

        assert_eq!(b.read_byte().unwrap(), 15);

        assert_eq!(
            b.read_byte().unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn read_bits() {
        let bytes = [0b01010111, 0b00011101, 0b11110101, 0b00010100];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.read_bits(3).unwrap(), 0b010);
        assert_eq!(b.read_bits(1).unwrap(), 0b1);
        assert_eq!(b.read_bits(20).unwrap(), 0b01110001110111110101);
        assert_eq!(b.read_bits(8).unwrap(), 0b00010100);
        assert_eq!(
            b.read_bits(4).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    #[allow(clippy::bool_assert_comparison)]
    fn read_mixed() {
        let bytes = [0b01101101, 0b01101101];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bits(3).unwrap(), 0b110);
        assert_eq!(b.read_byte().unwrap(), 0b11010110);
        assert_eq!(b.read_bits(2).unwrap(), 0b11);
        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bits(1).unwrap(), 0b1);
        assert_eq!(
            b.read_bit().unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn peak_bits() {
        let bytes = [0b01010111, 0b00011101, 0b11110101, 0b00010100];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.peak_bits(1).unwrap(), 0b0);
        assert_eq!(b.peak_bits(4).unwrap(), 0b0101);
        assert_eq!(b.peak_bits(8).unwrap(), 0b01010111);
        assert_eq!(b.peak_bits(20).unwrap(), 0b01010111000111011111);

        // read some individual bits we can test `peak_bits` when the position in the
        // byte we are currently reading is non-zero
        assert_eq!(b.read_bits(12).unwrap(), 0b010101110001);

        assert_eq!(b.peak_bits(1).unwrap(), 0b1);
        assert_eq!(b.peak_bits(4).unwrap(), 0b1101);
        assert_eq!(b.peak_bits(8).unwrap(), 0b11011111);
        assert_eq!(b.peak_bits(20).unwrap(), 0b11011111010100010100);

        assert_eq!(
            b.peak_bits(22).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    #[allow(clippy::bool_assert_comparison)]
    #[allow(clippy::unusual_byte_groupings)]
    fn read_many_bits() {
        let bytes = [
            0b0100_1001,
            0b1101_1011,
            0b1111_0101,
            0b0001_0100,
            0b1001_0110,
            0b1001_0101,
            0b1011_1101,
            0b1001_1011,
        ];
        let mut b = BitReader::new(&bytes[..]);

        assert_eq!(b.read_bit().unwrap(), false);
        assert_eq!(b.read_bit().unwrap(), true);
        // 6 + 8 * 5 + 6 = 52
        assert_eq!(
            b.read_bits(52).unwrap(),
            0b00_1001__1101_1011__1111_0101__0001_0100__1001_0110__1001_0101__1011_11
        );
    }

    #[test]
    #[allow(clippy::bool_assert_comparison)]
    #[allow(clippy::unusual_byte_groupings)]
    fn read_many_bits2() {
        let mut writer = BufferedBitWriter::new();
        writer.write_bits(0b0101_010, 63);
        writer.write_bits(0b1_0111, 5);
        writer.write_bits(0b111_1010, 7);

        let bytes = writer.as_slice();

        let mut reader = BitReader::new(bytes);

        assert_eq!(reader.read_bits(63 - 7).unwrap(), 0);
        assert_eq!(reader.read_bits(7).unwrap(), 0b0101_010);
        assert_eq!(reader.read_bits(5).unwrap(), 0b1_0111);
        assert_eq!(reader.read_bits(7).unwrap(), 0b111_1010);
    }
}
