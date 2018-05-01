require 'pg'

require 'pry'

class DBConnection
  attr_accessor :connect
  def initialize
    @connect = PG.connect(dbname: 'text_analyzer_dev')
  end



  def count
    sql = "SELECT count(phrase) FROM tokens;"
    result = connect.exec(sql)
    result.values.first.first.to_i
  end

  def [](category_name)
    sql = <<~SQL
    SELECT sum(frequency) AS total_tokens, count(phrase) AS examples
    FROM tokens
    JOIN category ON category.id = category_id
    WHERE category.name = $1;
    SQL
    result = connect.exec_params(sql, [category_name])
    # binding.pry
    total_tokens = result.values[0][0].to_i
    examples = result.values[0][1].to_i
    { tokens: create_token_hash(category_name),
      total_tokens: total_tokens,
      examples: examples }
  end

  def []=(phrase, frequency)
    sql = "UPDATE tokens SET frequency = $2 WHERE phrase = $1"
    result = connect.exec_params(sql, [phrase, frequency])
  end

  def output_all #debug output
    sql = "SELECT * FROM tokens;"
    result = connect.exec(sql)
    result.values
  end

  def create_token_hash(category)
    sql = <<~SQL
    SELECT phrase, frequency
    FROM tokens
    WHERE category_id = (SELECT id FROM category WHERE name = $1);
    SQL
    token_hash = {}
    result = connect.exec_params(sql, [category])
    result.map do |tuple|
      token_hash[tuple['phrase']] = tuple['frequency'].to_i
    end
    token_hash
  end

  # private

  # def connection
  #   self.connect = PG.connect(dbname: 'text_analyzer_dev')
  #   yield if block_given?
  #   self.connect.finish
  # end
end

class DBToken < DBConnect
  def keys
    sql = "SELECT phrase FROM tokens;"
    result = connect.exec(sql)
    result.field_values('phrase')
  end
end

class DBData < DBConnect
  def keys
    sql = "SELECT name FROM category;"
    result = connect.exec(sql)
    result.field_values('name')
  end


  def has_key?(value)
    sql = "SELECT name FROM category WHERE name = $1;"
    result = connect.exec_params(sql, [value])
    
  end
end
