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

pub struct BitReader<R: Read> {
    /// The underlying reader.
    reader: R,
    /// The buffer of bits read from the reader.
    ///
    /// The buffer bits are stored in `buffer_pos`...`buffer_pos + buffer_len` bits.
    buffer: u64,
    /// The number of bits in the buffer. 0 indicates that the buffer is empty.
    buffer_len: u8,
    /// The position of the first bit in the buffer.
    buffer_pos: u8,
}

impl<R: Read> BitReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            buffer_len: 0,
            buffer_pos: 0,
        }
    }

    /// Fill the buffer with 8 bits from the reader.
    /// Return the number of bytes read.
    ///
    /// The bits left in the buffer will be discarded.
    #[inline]
    fn fill_buffer_full(&mut self) -> io::Result<u8> {
        debug_assert!(self.buffer_len == 0);

        let mut byte = [0u8; 8];
        let n = self.reader.read(&mut byte)? as u8;

        self.buffer = u64::from_be_bytes(byte);

        self.buffer_len = n * 8;
        self.buffer_pos = 0;
        Ok(n)
    }

    /// Fill the buffer with 8 bits from the reader.
    /// Return the number of bytes read.
    ///
    /// The bits left in the buffer will be preserved.
    #[inline]
    fn fill_buffer(&mut self) -> io::Result<u8> {
        // reset buffer_pos
        self.buffer <<= self.buffer_pos;
        self.buffer_pos = 0;

        // The number of bytes to read.
        let mut n = (64 - self.buffer_len) / 8;
        debug_assert!(n * 8 + self.buffer_len <= 64);
        let mut byte = [0u8; 8];
        n = self.reader.read(&mut byte[0..(n as usize)])? as u8;
        let byte = u64::from_be_bytes(byte);
        println!("byte: \t\t\t\t\t{:064b}", byte);
        self.buffer = byte | (self.buffer >> (n * 8));
        self.buffer_len += 8 * n;
        Ok(n)
    }

    #[inline]
    fn read_bit_filled(&mut self) -> bool {
        let bit = (self.buffer >> (63 - self.buffer_pos)) & 1;
        self.buffer_pos += 1;
        self.buffer_len -= 1;
        bit == 1
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
        if self.buffer_len == 0 {
            let n_bits_read = self.fill_buffer_full()?;

            if n_bits_read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "EOF while reading a bit",
                ));
            }
        }
        let bit = (self.buffer >> (63 - self.buffer_pos)) & 1;
        self.buffer_pos += 1;
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
        let mut byte = 0u8;
        if self.buffer_len == 0 {
            self.fill_buffer_full()?;
        }
        if self.buffer_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF while reading a byte",
            ));
        }
        if self.buffer_pos % 8 != 0 {
            for _ in 0..8 {
                byte = (byte << 1) | (self.read_bit_filled() as u8);
            }
        } else {
            byte = (self.buffer >> (64 - 8 - self.buffer_pos)) as u8;
            self.buffer <<= 8;
            self.buffer_len -= 8;
        }
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
        if n > 64 {
            n = 64;
        }
        let mut bits = 0u64;

        while n > 8 {
            bits = (bits << 8) | self.read_byte()? as u64;
            n -= 8;
        }

        for _ in 0..n {
            bits = (bits << 1) | (self.read_bit()? as u64);
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
        if self.buffer_len < n {
            self.fill_buffer()?;
        }
        if self.buffer_len < n {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF while reading bits",
            ));
        }

        let bits = (self.buffer << self.buffer_pos) >> (64 - n);

        Ok(bits)
    }
}

#[cfg(test)]
mod tests {
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
}
