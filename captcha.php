<?php
require_once('vendor/autoload.php');

use Gregwar\Captcha\CaptchaBuilder;
$builder = new CaptchaBuilder;
$builder->build(200, 80);
$text = $builder->getPhrase();
header('Content-type: image/jpeg');
header('X-Phrase: ' . $text);
$builder->output();
//$builder->save("samples/$text.jpg");
